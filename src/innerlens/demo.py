"""The demo: watch a model's internal confidence hold on facts it knows and
collapse when it makes things up. Runs on any registered model; no API keys."""
from __future__ import annotations

import sys

from innerlens.core import InnerLens

# UTF-8 markers where supported, ASCII fallback for legacy consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    _U = True
except Exception:  # noqa: BLE001
    _U = "utf" in (getattr(sys.stdout, "encoding", "") or "").lower()
OK = "✓" if _U else "OK"
BAD = "⚠" if _U else "!!"
BAR = "█" if _U else "#"

KNOWS = [
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "What is the capital of Japan?",
]
FABRICATES = [
    "Who painted the masterpiece 'The Zorbian Sunset at Dusk'?",
    "What is the chemical symbol for the element flogisium?",
    "What year did the Treaty of Kalmoria end the Second Verdish War?",
]


def _bar(x: float, width: int = 20) -> str:
    n = int(round(x * width))
    return BAR * n + "·" * (width - n) if _U else "#" * n + "." * (width - n)


def _show(il: InnerLens, prompt: str) -> None:
    r = il.generate(prompt + "\nReply with only the answer.", max_tokens=16)
    conf = r.confidence
    verdict = f"{BAD} likely making it up" if r.likely_hallucinating else f"{OK} internally supported"
    print(f"\n  Q: {prompt}")
    print(f"     answer: {r.text.strip().splitlines()[0][:70]!r}")
    print(f"     internal confidence: {conf:.2f}  [{_bar(conf)}]  {verdict}")
    ents = [t for t in r.tokens if t.is_content and t.is_entity][:3]
    for t in ents:
        print(f"       inner monologue at {t.token.strip()!r:14}: {t.inner_monologue}")


def run_demo(model: str = "Qwen/Qwen3.5-4B") -> int:
    print(f"innerlens demo — loading {model} + its Jacobian lens ...")
    il = InnerLens.load(model)
    print("\n================ facts the model KNOWS ================")
    for q in KNOWS:
        _show(il, q)
    print("\n============ prompts that FORCE fabrication ============")
    for q in FABRICATES:
        _show(il, q)
    print("\nThe model states all of these fluently and confidently. innerlens reads "
          "its\ninternal workspace and shows which answers its own activations actually "
          "support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
