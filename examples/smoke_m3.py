from innerlens import InnerLens

il = InnerLens.load("Qwen/Qwen3.5-4B")
tests = [
    "Who painted the Mona Lisa?",
    "What is the capital of Australia?",
    "Who painted the artwork 'The Zorbian Sunset at Dusk'?",
    "What is the middle name of the 12th mayor of Brindlewick, Ohio?",
]
for p in tests:
    r = il.generate(p, max_tokens=12)
    flag = "  <-- FLAGGED (likely making it up)" if r.likely_hallucinating else ""
    print(f"\nQ: {p}")
    print(f"  answer: {r.text.strip()!r}")
    print(f"  confidence={r.confidence:.3f}  min={r.min_confidence:.3f}{flag}")
    for t in r.tokens[:6]:
        if t.is_content:
            print(f"    token {t.token!r:14} internal_conf={t.internal_confidence:.3f} "
                  f"inner_monologue={t.inner_monologue}")
