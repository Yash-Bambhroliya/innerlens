# Changelog

## [0.1.0] — unreleased

First release: read a model's internal workspace to see what it's really
thinking, and whether it's making it up.

### Added
- `InnerLens.load(model)` + `generate_with_workspace()` — per-token
  internal-confidence, workspace entropy, and inner-monologue via the Jacobian
  lens; a `likely_hallucinating` flag and an entity-level confidence score.
- Pre-fitted lens registry (Qwen3.5-4B, Qwen3.6-27B).
- OpenAI-compatible server (`innerlens serve`): drop-in `/v1/chat/completions`
  that adds an `x_workspace` block (confidence + per-token trace) to every
  response. Validated with the real `openai` Python SDK.
- `innerlens demo` — offline introspection demo (truthful vs fabricated).
- Validation: premise (internal-confidence 0.99 truthful vs 0.19 fabricated),
  hallucination AUROC 0.80 on TriviaQA, 9 scoring unit tests. See RESULTS.md.

### Credit
Built on anthropics/jacobian-lens (Apache-2.0) and the Neuronpedia pre-fitted
lenses. Apache-2.0.

[0.1.0]: https://github.com/innerlens/innerlens/releases/tag/v0.1.0
