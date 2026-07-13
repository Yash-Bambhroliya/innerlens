"""Registry of pre-fitted Jacobian lenses for open models.

Lenses are published on HuggingFace by the Neuronpedia / Anthropic release
(companion to "A global workspace in language models"). innerlens maps a model
name to its lens so `InnerLens.load("Qwen/Qwen3.5-4B")` just works; pass an
explicit `lens=` to override, or fit your own with jlens.fit.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LensRef:
    repo: str
    filename: str
    revision: str


# Verified against the jacobian-lens walkthrough (neuronpedia/jacobian-lens).
REGISTRY: dict[str, LensRef] = {
    "Qwen/Qwen3.5-4B": LensRef(
        "neuronpedia/jacobian-lens",
        "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
        "qwen-n1000",
    ),
    "Qwen/Qwen3.6-27B": LensRef(
        "neuronpedia/jacobian-lens",
        "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
        "qwen-n1000",
    ),
}


def lens_for(model_name: str) -> LensRef:
    if model_name not in REGISTRY:
        raise KeyError(
            f"No pre-fitted lens registered for {model_name!r}. "
            f"Known: {sorted(REGISTRY)}. Pass lens=LensRef(...) to use your own, "
            f"or fit one with jlens.fit (see docs).")
    return REGISTRY[model_name]
