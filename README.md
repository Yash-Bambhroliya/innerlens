# innerlens

**See what your open model is really thinking — and whether it's making it up.**

When an LLM answers, it sounds equally confident whether it *knows* the fact or
is *fabricating* one. innerlens reads the model's **internal workspace** (via
Anthropic's Jacobian lens) while it generates, and for every token shows what the
model was internally *disposed to say* — its inner monologue — and how strongly
its own activations actually support the token it produced. When the model states
a real fact, internal support is high; when it makes something up, the support
collapses.

It's a **drop-in**: point any OpenAI SDK at the innerlens server and every
response comes back with a hallucination signal and a per-token introspection
trace — no second model, no extra API calls, no code changes beyond the base URL.

[![PyPI](https://img.shields.io/pypi/v/innerlens.svg)](https://pypi.org/project/innerlens/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Built on jacobian-lens](https://img.shields.io/badge/built%20on-anthropics%2Fjacobian--lens-black.svg)](https://github.com/anthropics/jacobian-lens)

---

## Watch it happen (real output, Qwen3.5-4B)

```text
================ facts the model KNOWS ================
  Q: What is the chemical symbol for gold?
     answer: 'Au'
     internal confidence: 0.98  [████████████████████]  ✓ internally supported
       inner monologue at 'Au': ['Au', ' Au', 'Ag', 'AU']       # even considers Ag (silver!)

  Q: What is the capital of Japan?
     answer: 'Tokyo'
     internal confidence: 0.97  [███████████████████·]  ✓ internally supported
       inner monologue at 'Tok': ['Tok', ' Tokyo', '东京', ' Tok']   # thinking in 日本語 too

============ prompts that FORCE fabrication ============
  Q: What year did the Treaty of Kalmoria end the Second Verdish War?
     answer: '1920'
     internal confidence: 0.17  [███·················]  ⚠ likely making it up
       inner monologue at '1': ['1','2','There','4']
       inner monologue at '9': ['9','8','7','4']     # internals just cycling random digits —
       inner monologue at '2': ['9','4','2','8']     # the model has no idea, but says "1920"
```

The treaty is fictional. The model answers fluently anyway — but its internal
workspace is visibly guessing, and innerlens catches it.

## Install

```bash
pip install innerlens                 # the library
pip install "innerlens[runtime]"      # torch, transformers, accelerate
pip install "git+https://github.com/anthropics/jacobian-lens"   # the lens lib (GitHub-only)
```

Runs any HuggingFace decoder with a published Jacobian lens (Qwen3.5-4B and
Qwen3.6-27B ship in the registry; bring your own with `jlens.fit`). A GPU is
recommended.

## Use it as a library

```python
from innerlens import InnerLens

il = InnerLens.load("Qwen/Qwen3.5-4B")     # loads the model + its lens
r = il.generate("Who painted the Mona Lisa?")

print(r.text)                  # "Leonardo da Vinci"
print(r.confidence)            # 0.71  — internal support for the weakest answer token
print(r.likely_hallucinating)  # False
for t in r.tokens:             # per-token introspection
    print(t.token, t.internal_confidence, t.inner_monologue)
```

## Use it as a drop-in OpenAI endpoint (the wedge)

```bash
innerlens serve --model Qwen/Qwen3.5-4B    # OpenAI-compatible server on :8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

r = client.chat.completions.create(model="innerlens",
        messages=[{"role": "user", "content": "What is the capital of Japan?"}])

r.choices[0].message.content          # "Tokyo"
r.x_workspace["confidence"]           # 0.97
r.x_workspace["likely_hallucinating"] # False   <- every response, for free
```

Your app doesn't change. `x_workspace` is an extra field OpenAI SDKs ignore, so
existing code keeps working while new code can read the signal.

## Does the hallucination signal actually work?

Yes — measured, not asserted. On **TriviaQA** (200 questions, Qwen3.5-4B), the
internal-confidence signal predicts whether the model's answer is **correct**:

| signal | AUROC |
| --- | --- |
| innerlens internal-confidence (weakest entity token) | **0.80** |
| mean entity-token confidence | 0.75 |

That's a free, single-forward-pass signal read from the model's own internals —
no second model, no sampling, no external knowledge base. Reproduce it with
[`eval_via_library.py`](eval_via_library.py).

**Honest caveats.** This is an early result on one 4B model and one dataset, and
the signal is imperfect (a confident model can still be wrong, and a correct
refusal can read low). String-match grading is strict, so the true AUROC is
likely a touch higher than measured. Treat the score as a **strong prior, not a
verdict** — and see [what it does and doesn't catch](#how-it-works).

## How it works

The Jacobian lens (Anthropic, ["A global workspace in language
models"](https://www.anthropic.com/research/global-workspace)) transports a
mid-network activation into the model's output vocabulary:
`lens_l(h) = unembed(J_l @ h)`, reading *what that activation is disposed to make
the model say*. innerlens reads this at the model's late layers for each generated
token and reports:

- **`internal_confidence`** — how strongly the late-layer workspace supports the
  token the model actually emitted. High on known facts, low on fabrications.
- **`inner_monologue`** — the top internal dispositions (what it was "thinking").
- The **hallucination score** aggregates internal-confidence over the answer's
  entity tokens (skipping function words and trailing rambling).

It reads a signal the model can't fake because it isn't in the output text — it's
in the activations underneath.

## Limitations

- Needs a fitted lens for the model (registry ships Qwen; others via `jlens.fit`).
- v1 computes the per-token readout with one forward per traced token — fine for
  short answers/demos; batched activation-hook streaming is on the roadmap.
- Single-machine, greedy decoding in v1.
- The signal is a research-grade early result, not a certified safety guarantee.

## Credit & license

Built on [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)
(Apache-2.0) and the pre-fitted lenses released via Neuronpedia. innerlens is
Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
