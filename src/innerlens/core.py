"""innerlens core — read a model's internal workspace while it generates.

For every token the model emits, innerlens uses the Jacobian lens (jlens) to
read what the model was *internally disposed to say* at that point — its inner
monologue — and how strongly its own internals supported the token it actually
produced. When a model states a known fact, internal support is high; when it
fabricates, the support collapses and the workspace goes high-entropy (the
premise verified on Qwen3.5-4B: internal-confidence 0.99 truthful vs 0.19
fabricated; predicts answer correctness on TriviaQA at AUROC ~0.78).

This module is the engine; `server.py` exposes it as an OpenAI-compatible API.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from innerlens.lenses import LensRef, lens_for

# torch / transformers / jlens are imported lazily inside load() so that
# `import innerlens` is cheap and never fights a user's CUDA/torch install.

DEFAULT_HALLUCINATION_THRESHOLD = 0.5  # entity-token internal-confidence below this = flag

# Discourse / function words carry little factual commitment, and (being high-
# frequency, low-information) tend to have low internal-confidence regardless of
# truth. The hallucination score is computed over ENTITY tokens (content minus
# these), so a correct "The capital of Australia is Canberra" isn't dragged down
# by its leading "The".
_STOPWORDS = frozenset("""
a an the this that these those it its it's there their his her our your my
is are was were be been being am do does did has have had will would can could
of to in on at for and or but nor so as by with from into onto about over under
no not yes well actually based here what which who whom whose when where why how
i you he she we they them us me him
""".split())


@dataclass
class TokenTrace:
    """The workspace readout for one generated token."""
    token: str
    token_id: int
    internal_confidence: float          # late-layer J-lens support for this token [0,1]
    entropy: float                      # workspace entropy (nats); high = uncertain
    inner_monologue: List[str] = field(default_factory=list)  # top internal dispositions
    is_content: bool = True             # False for whitespace/punctuation tokens
    is_entity: bool = True              # False for stopwords/discourse markers too


@dataclass
class WorkspaceResult:
    text: str
    tokens: List[TokenTrace]
    threshold: float = DEFAULT_HALLUCINATION_THRESHOLD

    @property
    def _entities(self) -> List[TokenTrace]:
        # Score the answer's first sentence only — trailing rambling after the
        # answer (e.g. "...Canberra.\n\nLocated in...") shouldn't drag the score.
        ents: List[TokenTrace] = []
        for t in self.tokens:
            if t.is_content and t.is_entity:
                ents.append(t)
            if ents and ("\n" in t.token or t.token.strip().endswith((".", "!", "?"))):
                break
        if ents:
            return ents
        return [t for t in self.tokens if t.is_content] or self.tokens

    @property
    def confidence(self) -> float:
        """Headline signal: the weakest-internally-supported ENTITY token in the
        answer — the fabrication tell. A correct answer supports all its
        entities; a made-up one has at least one unsupported entity."""
        return min((t.internal_confidence for t in self._entities), default=1.0)

    @property
    def min_confidence(self) -> float:
        return self.confidence

    @property
    def mean_entity_confidence(self) -> float:
        ents = self._entities
        return sum(t.internal_confidence for t in ents) / len(ents) if ents else 1.0

    @property
    def likely_hallucinating(self) -> bool:
        return self.confidence < self.threshold

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "min_confidence": round(self.min_confidence, 4),
            "likely_hallucinating": self.likely_hallucinating,
            "tokens": [
                {"token": t.token, "internal_confidence": round(t.internal_confidence, 4),
                 "entropy": round(t.entropy, 4), "inner_monologue": t.inner_monologue}
                for t in self.tokens
            ],
        }


class InnerLens:
    """Wraps a HuggingFace causal LM + its Jacobian lens, and exposes the
    internal workspace during generation."""

    def __init__(self, hf_model, tokenizer, lens_model, lens, late_layers: Sequence[int],
                 model_name: str, monologue_k: int = 4):
        self.hf = hf_model
        self.tok = tokenizer
        self.model = lens_model           # jlens HFLensModel
        self.lens = lens                  # jlens JacobianLens
        self.late_layers = list(late_layers)
        self.model_name = model_name
        self.monologue_k = monologue_k

    @classmethod
    def load(cls, model_name: str = "Qwen/Qwen3.5-4B", *, lens: Optional[LensRef] = None,
             device: str = "cuda", dtype: str = "bfloat16", monologue_k: int = 4) -> "InnerLens":
        import torch
        import transformers
        import jlens

        jlens.configure_logging()
        ref = lens or lens_for(model_name)
        torch_dtype = getattr(torch, dtype)
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype).to(device)
        tok = transformers.AutoTokenizer.from_pretrained(model_name)
        lens_model = jlens.from_hf(hf, tok)
        loaded_lens = jlens.JacobianLens.from_pretrained(
            ref.repo, filename=ref.filename, revision=ref.revision)
        n = lens_model.n_layers
        late = [n * 3 // 4, n - 2]        # layers where the readout is linguistically meaningful
        return cls(hf, tok, lens_model, loaded_lens, late, model_name, monologue_k)

    # -- prompt formatting -------------------------------------------------- #
    def _format(self, prompt: str, chat: bool, enable_thinking: bool) -> str:
        if not chat:
            return prompt
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except TypeError:
            # tokenizers without the enable_thinking kwarg
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def format_messages(self, messages, enable_thinking: bool = False) -> str:
        """Render an OpenAI-style messages list to a prompt via the chat template."""
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except TypeError:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    # -- the workspace read ------------------------------------------------- #
    def _readout(self, context_text: str, token_id: int):
        """Internal-confidence, entropy, and inner-monologue for `token_id`
        given the disposition at the end of context_text."""
        import torch
        jl, _, _ = self.lens.apply(self.model, context_text, layers=self.late_layers,
                                   positions=[-1])
        best_conf, best_ent, monologue = 0.0, 0.0, []
        for layer in self.late_layers:
            logits = jl[layer][0].float()
            p = torch.softmax(logits, -1)
            conf = p[token_id].item()
            if conf >= best_conf:
                best_conf = conf
                best_ent = -(p * torch.log(p + 1e-9)).sum().item()
                monologue = [self.tok.decode([t]) for t in logits.topk(self.monologue_k).indices]
        return best_conf, best_ent, monologue

    @staticmethod
    def _is_content(piece: str) -> bool:
        s = piece.strip()
        return bool(s) and not all(not c.isalnum() for c in s)

    @staticmethod
    def _is_entity(piece: str) -> bool:
        s = piece.strip().lower().strip(".,;:!?'\"()[]")
        if not s or all(not c.isalnum() for c in s):
            return False
        return s not in _STOPWORDS

    def generate_with_workspace(self, prompt: str, *, max_tokens: int = 32,
                                chat: bool = True, enable_thinking: bool = False,
                                trace_tokens: Optional[int] = None,
                                threshold: float = DEFAULT_HALLUCINATION_THRESHOLD
                                ) -> WorkspaceResult:
        """Greedily generate, and for each generated token read the internal
        workspace. `trace_tokens` caps how many leading tokens get a (costly)
        lens read — enough to cover the answer's commitment; the rest stream
        without a readout."""
        import torch
        formatted = self._format(prompt, chat, enable_thinking)
        input_ids = self.tok(formatted, return_tensors="pt").input_ids.to(self.hf.device)
        with torch.no_grad():
            out = self.hf.generate(input_ids, max_new_tokens=max_tokens, do_sample=False,
                                   pad_token_id=self.tok.eos_token_id)
        gen_ids = out[0, input_ids.shape[1]:].tolist()
        text = self.tok.decode(gen_ids, skip_special_tokens=True)

        cap = trace_tokens if trace_tokens is not None else max_tokens
        traces: List[TokenTrace] = []
        for i, tid in enumerate(gen_ids):
            if tid == self.tok.eos_token_id:
                break
            piece = self.tok.decode([tid])
            if i < cap:
                ctx = formatted + self.tok.decode(gen_ids[:i])
                conf, ent, mono = self._readout(ctx, tid)
            else:
                conf, ent, mono = float("nan"), float("nan"), []
            traces.append(TokenTrace(
                token=piece, token_id=tid, internal_confidence=conf, entropy=ent,
                inner_monologue=mono, is_content=self._is_content(piece),
                is_entity=self._is_entity(piece)))
        return WorkspaceResult(text=text, tokens=traces, threshold=threshold)

    # convenience alias
    generate = generate_with_workspace
