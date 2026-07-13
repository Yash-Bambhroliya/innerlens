# Validation results (Qwen3.5-4B, H100)

Every claim in the README was validated on real hardware (NVIDIA H100, Qwen3.5-4B
+ the pre-fitted `neuronpedia/jacobian-lens` `qwen-n1000` lens).

## 1. Premise — internal support separates truth from fabrication

Reading the late-layer J-lens support for the emitted next token:

| group | mean internal-confidence | mean workspace entropy |
| --- | --- | --- |
| truthful facts | **0.99** | 0.07 |
| forced fabrications | **0.19** | 3.51 |

Two of the fabrication prompts contained no "fictional" cue (genuine unknowns) and
the signal still fired — it is internal uncertainty, not keyword detection.

## 2. Hallucination signal — measured, not asserted

TriviaQA (rc.nocontext, 200 questions), scored through the actual library
(`WorkspaceResult.confidence`, reproduced by `eval_via_library.py`):

- model accuracy 39% (78 correct / 122 wrong — a genuine mix)
- **AUROC 0.80** — internal-confidence (weakest entity token) predicts answer correctness
- AUROC 0.75 — mean entity-token confidence

A free, single-forward-pass signal from the model's own internals. Honest caveats
in the README: early result, one 4B model, one dataset, strict grading (true AUROC
likely a touch higher).

## 2b. Honest baseline — is it just output-token probability? (No — it's weaker as a scalar)

We benchmarked the internal signal against the model's own output-token
probability, read at the **identical tokens** from the **same lens call**, with
identical aggregation and grading (`eval/eval_baseline.py`, n=500 per dataset):

| signal | TriviaQA | PopQA |
| --- | --- | --- |
| internal-confidence (weakest entity) | 0.794 | 0.615 |
| output-token probability (weakest entity) | **0.857** | 0.585 |
| sequence mean log-prob | **0.859** | 0.604 |

Output-prob wins significantly on TriviaQA (DeLong p<0.0001); PopQA is a
statistical tie (p=0.14, and its strict grading is label-noisy). Combining the
signals adds nothing over output-prob alone (0.848 vs 0.849). Spearman rho
between the signals: 0.78 / 0.70 — correlated, not identical.

**Published conclusion:** innerlens's internal-confidence replicates (0.794 at
n=500 vs 0.80 at n=200) but is NOT a better bare scalar than the model's own
token probability. The unique value is the per-token *trace* (inner monologue +
internal support) and the drop-in surface. The README says exactly this.

## 3. Drop-in server — validated with the real OpenAI SDK

Pointing the `openai` Python client at the innerlens server (only `base_url`
changed):

```
Q: "What is the capital of Japan?"           -> "Tokyo",  x_workspace.confidence=0.97, likely_hallucinating=False
Q: "...Treaty of Kalmoria / 2nd Verdish War" -> "1914",   x_workspace.confidence=0.03, likely_hallucinating=True
```

## 4. Scoring logic — 9 unit tests, no GPU (`tests/test_scoring.py`)

Entity aggregation, first-sentence truncation, threshold flagging, registry.
