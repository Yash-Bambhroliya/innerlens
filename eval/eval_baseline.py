"""Is innerlens's internal-confidence just output-token probability? Measured.

For every generated token, ONE `lens.apply` call yields both signals at the
identical position:
  - internal_confidence = softmax(jl[L])[token_id]      (late-layer J-lens; ours)
  - output_confidence   = softmax(model_logits)[token_id]  (the model's own softmax)
Token selection, entity aggregation (via the actual `WorkspaceResult` code path),
grading, and prompts are held identical — the only difference is the signal.

Also reports the other standard token-prob baselines (length-normalized sequence
log-prob, predictive entropy at the first entity token), bootstrap 95% CIs, a
paired DeLong test, Spearman overlap, and a combined-signal AUROC.

Usage (on the GPU box):
    python eval/eval_baseline.py --dataset trivia --n 500
    python eval/eval_baseline.py --dataset popqa  --n 500
"""
import argparse
import json
import math
import re
import string

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from innerlens import InnerLens
from innerlens.core import TokenTrace, WorkspaceResult

_printed_shapes = False


# ---------------------------------------------------------------- grading -- #
def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def load_examples(name, n):
    if name == "trivia":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                          split=f"validation[:{n}]")
        for ex in ds:
            gold = [ex["answer"]["value"]] + list(ex["answer"].get("aliases", []))
            yield ex["question"], gold
    elif name == "popqa":
        ds = load_dataset("akariasai/PopQA", split=f"test[:{n}]")
        for ex in ds:
            raw = ex["possible_answers"]
            try:
                gold = list(json.loads(raw))
            except Exception:
                import ast
                gold = list(ast.literal_eval(raw))
            yield ex["question"], [str(g) for g in gold]
    else:
        raise ValueError(name)


# ------------------------------------------------------------ measurement -- #
def measure(il, prompt, max_tokens=16):
    """Replicates InnerLens.generate_with_workspace's per-token loop, but reads
    BOTH the J-lens internal-confidence and the model's output-token probability
    from the same lens.apply call at the same position."""
    global _printed_shapes
    formatted = il._format(prompt, chat=True, enable_thinking=False)
    input_ids = il.tok(formatted, return_tensors="pt").input_ids.to(il.hf.device)
    with torch.no_grad():
        out = il.hf.generate(input_ids, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=il.tok.eos_token_id)
    gen_ids = out[0, input_ids.shape[1]:].tolist()
    text = il.tok.decode(gen_ids, skip_special_tokens=True)

    tr_int, tr_out = [], []
    logprobs = []
    first_entity_entropy = None
    for i, tid in enumerate(gen_ids):
        if tid == il.tok.eos_token_id:
            break
        piece = il.tok.decode([tid])
        ctx = formatted + il.tok.decode(gen_ids[:i])
        jl, model_logits, _ = il.lens.apply(il.model, ctx, layers=il.late_layers,
                                            positions=[-1])
        if not _printed_shapes:
            print(f"[shapes] jl[{il.late_layers[0]}]: {tuple(jl[il.late_layers[0]].shape)}  "
                  f"model_logits: {type(model_logits).__name__} "
                  f"{tuple(torch.as_tensor(model_logits).shape)}", flush=True)
            _printed_shapes = True

        # internal-confidence: identical to InnerLens._readout (max over late layers)
        int_conf = 0.0
        for layer in il.late_layers:
            p = torch.softmax(jl[layer][0].float(), -1)
            int_conf = max(int_conf, p[tid].item())

        # output-token probability at the SAME position
        ml = torch.as_tensor(model_logits)
        while ml.dim() > 1:
            ml = ml[-1]
        q = torch.softmax(ml.float(), -1)
        out_conf = q[tid].item()
        logprobs.append(math.log(max(out_conf, 1e-12)))

        is_c, is_e = InnerLens._is_content(piece), InnerLens._is_entity(piece)
        if first_entity_entropy is None and is_c and is_e:
            first_entity_entropy = -(q * torch.log(q + 1e-9)).sum().item()

        kw = dict(token=piece, token_id=tid, entropy=0.0, inner_monologue=[],
                  is_content=is_c, is_entity=is_e)
        tr_int.append(TokenTrace(internal_confidence=int_conf, **kw))
        tr_out.append(TokenTrace(internal_confidence=out_conf, **kw))

    # identical aggregation code path (entity tokens, first sentence) for both
    ws_int = WorkspaceResult(text=text, tokens=tr_int)
    ws_out = WorkspaceResult(text=text, tokens=tr_out)
    return {
        "text": text,
        "internal_min": ws_int.confidence,
        "internal_mean": ws_int.mean_entity_confidence,
        "output_min": ws_out.confidence,
        "output_mean": ws_out.mean_entity_confidence,
        "seq_logprob": (sum(logprobs) / len(logprobs)) if logprobs else 0.0,
        "neg_entropy": -(first_entity_entropy if first_entity_entropy is not None else 10.0),
    }


# ------------------------------------------------------------------ stats -- #
def bootstrap_auroc(y, s, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return roc_auc_score(y, s), lo, hi


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out


def delong_test(y, s1, s2):
    """Paired DeLong test for two correlated AUROCs. Returns (auc1, auc2, z, p)."""
    y = np.asarray(y)
    order = np.argsort(-y)
    preds = np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]])
    m = int(y.sum())
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / max(math.sqrt(max(var, 0.0)), 1e-12)
    p = math.erfc(abs(z) / math.sqrt(2))
    return aucs[0], aucs[1], z, p


def spearman(a, b):
    ra, rb = _midrank(np.asarray(a)), _midrank(np.asarray(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def combo_auroc(y, feats, seed=0):
    """Out-of-fold logistic-regression AUROC on the given feature columns."""
    X = np.column_stack(feats)
    y = np.asarray(y)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


# ---------------------------------------------------------------- analyze -- #
SIGNALS = [
    ("internal_min", "innerlens internal-confidence (weakest entity)"),
    ("internal_mean", "innerlens internal-confidence (mean entity)"),
    ("output_min", "output-token probability (weakest entity)"),
    ("output_mean", "output-token probability (mean entity)"),
    ("seq_logprob", "sequence mean log-prob (perplexity baseline)"),
    ("neg_entropy", "neg. predictive entropy @ first entity"),
]


def analyze(rows, name):
    y = np.array([r["correct"] for r in rows])
    print(f"\n================ {name}  n={len(y)}  accuracy={y.mean():.1%} "
          f"(correct={y.sum()}, wrong={len(y) - y.sum()}) ================")
    if y.sum() in (0, len(y)):
        print("degenerate labels; cannot compute AUROC")
        return

    cols = {k: np.array([r[k] for r in rows]) for k, _ in SIGNALS}
    print(f"\n{'signal':52s} {'AUROC':>6s}  {'95% CI':>16s}")
    for k, label in SIGNALS:
        auc, lo, hi = bootstrap_auroc(y, cols[k])
        print(f"{label:52s} {auc:6.3f}  [{lo:.3f}, {hi:.3f}]")

    a1, a2, z, p = delong_test(y, cols["internal_min"], cols["output_min"])
    print(f"\nDeLong paired (internal_min vs output_min): "
          f"{a1:.3f} vs {a2:.3f}  delta={a1 - a2:+.3f}  z={z:+.2f}  p={p:.4f}")
    a1, a2, z, p = delong_test(y, cols["internal_mean"], cols["output_mean"])
    print(f"DeLong paired (internal_mean vs output_mean): "
          f"{a1:.3f} vs {a2:.3f}  delta={a1 - a2:+.3f}  z={z:+.2f}  p={p:.4f}")

    rho = spearman(cols["internal_min"], cols["output_min"])
    print(f"\nSpearman rho (internal_min, output_min) = {rho:.3f}")

    combo = combo_auroc(y, [cols["internal_min"], cols["output_min"]])
    solo_out = combo_auroc(y, [cols["output_min"]])
    solo_int = combo_auroc(y, [cols["internal_min"]])
    print(f"combined LR AUROC (internal+output) = {combo:.3f}   "
          f"(output-only {solo_out:.3f}, internal-only {solo_int:.3f})")

    # confident-but-hollow: model's own softmax is confident, internals are not
    med_out = np.median(cols["output_min"])
    idx = [i for i in range(len(y))
           if cols["output_min"][i] >= med_out and cols["internal_min"][i] < 0.5]
    idx.sort(key=lambda i: cols["output_min"][i] - cols["internal_min"][i], reverse=True)
    wrong = sum(1 - y[i] for i in idx)
    print(f"\nconfident-but-hollow (output_min>=median({med_out:.2f}), internal_min<0.5): "
          f"{len(idx)} cases, {wrong} wrong "
          f"({wrong / len(idx):.0%} hallucination rate)" if idx else
          "\nconfident-but-hollow: none")
    for i in idx[:10]:
        r = rows[i]
        print(f"  [{'WRONG' if not y[i] else 'ok   '}] out={r['output_min']:.2f} "
              f"int={r['internal_min']:.2f}  Q: {r['question'][:70]!r} "
              f"-> {r['text'][:40]!r}")

    # the reverse cell, for completeness
    ridx = [i for i in range(len(y))
            if cols["internal_min"][i] >= 0.5 and cols["output_min"][i] < med_out]
    if ridx:
        rwrong = sum(1 - y[i] for i in ridx)
        print(f"reverse (internal high, output low): {len(ridx)} cases, "
              f"{rwrong} wrong ({rwrong / len(ridx):.0%})")


# ------------------------------------------------------------------- main -- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trivia", choices=["trivia", "popqa"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    il = InnerLens.load("Qwen/Qwen3.5-4B")
    rows = []
    for i, (q, gold) in enumerate(load_examples(args.dataset, args.n)):
        m = measure(il, f"{q}\nReply with only the answer, no explanation.",
                    args.max_tokens)
        na = norm(m["text"])
        m["question"] = q
        m["gold"] = gold[:5]
        m["correct"] = int(any(norm(g) and norm(g) in na for g in gold))
        rows.append(m)
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{args.n}] acc so far "
                  f"{np.mean([r['correct'] for r in rows]):.1%}", flush=True)

    out = args.out or f"results_{args.dataset}_{len(rows)}.json"
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"per-example records -> {out}")
    analyze(rows, args.dataset)


if __name__ == "__main__":
    main()
