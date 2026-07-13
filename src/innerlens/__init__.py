"""innerlens — see what your open model is really thinking, and whether it's
making it up.

    from innerlens import InnerLens
    il = InnerLens.load("Qwen/Qwen3.5-4B")
    r = il.generate("Who painted the Mona Lisa?")
    print(r.text, "| internal:", r.confidence, "| output-prob:", r.output_confidence,
          "| hallucinating:", r.likely_hallucinating)
    for t in r.tokens:
        print(t.token, t.internal_confidence, t.output_confidence, t.inner_monologue)
"""
from innerlens.core import (
    DEFAULT_HALLUCINATION_THRESHOLD,
    InnerLens,
    TokenTrace,
    WorkspaceResult,
)
from innerlens.lenses import LensRef, REGISTRY

__version__ = "0.2.0"
__all__ = [
    "InnerLens",
    "WorkspaceResult",
    "TokenTrace",
    "LensRef",
    "REGISTRY",
    "DEFAULT_HALLUCINATION_THRESHOLD",
    "__version__",
]
