from openai import OpenAI

c = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
for q in ["What is the capital of Japan? Reply with only the answer.",
          "What year did the Treaty of Kalmoria end the Second Verdish War? Reply with only the answer."]:
    r = c.chat.completions.create(model="innerlens",
                                  messages=[{"role": "user", "content": q}], max_tokens=16)
    xw = (r.model_extra or {}).get("x_workspace", {})
    print(f"  Q: {q[:45]}")
    print(f"     content: {r.choices[0].message.content.strip()!r}")
    print(f"     x_workspace.confidence={xw.get('confidence')}  "
          f"likely_hallucinating={xw.get('likely_hallucinating')}")
