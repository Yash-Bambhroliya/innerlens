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

## 3. Drop-in server — validated with the real OpenAI SDK

Pointing the `openai` Python client at the innerlens server (only `base_url`
changed):

```
Q: "What is the capital of Japan?"           -> "Tokyo",  x_workspace.confidence=0.97, likely_hallucinating=False
Q: "...Treaty of Kalmoria / 2nd Verdish War" -> "1914",   x_workspace.confidence=0.03, likely_hallucinating=True
```

## 4. Scoring logic — 9 unit tests, no GPU (`tests/test_scoring.py`)

Entity aggregation, first-sentence truncation, threshold flagging, registry.
