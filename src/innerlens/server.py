"""OpenAI-compatible server that adds the internal workspace to every response.

Point any OpenAI SDK at this server and every completion comes back with an
`x_workspace` block — the model's per-token internal-confidence, inner
monologue, and a hallucination flag — computed from the model's own internals,
no second model, no extra API calls. This is the drop-in wedge: existing apps
get introspection + a hallucination signal with zero code changes beyond the
base_url.

    innerlens serve --model Qwen/Qwen3.5-4B      # start it
    # then, in your app:
    #   client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
    #   r = client.chat.completions.create(model="innerlens", messages=[...])
    #   r.choices[0].message.content        # the answer
    #   r.x_workspace["confidence"]         # internal-confidence [0,1]
    #   r.x_workspace["likely_hallucinating"]
"""
from __future__ import annotations

import os
import time
import uuid
from typing import List, Optional

from pydantic import BaseModel

from innerlens.core import DEFAULT_HALLUCINATION_THRESHOLD, InnerLens

_STATE: dict = {}


# Module-level so Pydantic v2 can resolve the model references (locally-scoped
# models trigger "TypeAdapter not fully defined").
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "innerlens"
    messages: List[Message]
    max_tokens: int = 64
    temperature: float = 0.0  # accepted for compatibility; generation is greedy
    threshold: Optional[float] = None


def get_lens() -> InnerLens:
    if "il" not in _STATE:
        model = os.environ.get("INNERLENS_MODEL", "Qwen/Qwen3.5-4B")
        _STATE["il"] = InnerLens.load(model)
    return _STATE["il"]


def build_app(preload: bool = False):
    from fastapi import Body, FastAPI

    app = FastAPI(title="innerlens", description="OpenAI-compatible API with an "
                  "internal-workspace / hallucination signal on every response.")

    if preload:
        get_lens()

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded": "il" in _STATE}

    @app.get("/v1/models")
    def models():
        model = os.environ.get("INNERLENS_MODEL", "Qwen/Qwen3.5-4B")
        return {"object": "list", "data": [{"id": "innerlens", "object": "model",
                                            "owned_by": "innerlens", "backing_model": model}]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest = Body(...)):
        il = get_lens()
        threshold = req.threshold if req.threshold is not None else DEFAULT_HALLUCINATION_THRESHOLD
        formatted = il.format_messages(
            [{"role": m.role, "content": m.content} for m in req.messages])
        result = il.generate_with_workspace(
            formatted, max_tokens=req.max_tokens, chat=False, threshold=threshold)
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(result.tokens),
                      "total_tokens": len(result.tokens)},
            # the extension: unknown fields are ignored by OpenAI SDKs, so this
            # is a safe drop-in enrichment.
            "x_workspace": result.to_dict(),
        }

    return app


def serve(model: Optional[str] = None, host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    if model:
        os.environ["INNERLENS_MODEL"] = model
    app = build_app(preload=True)
    uvicorn.run(app, host=host, port=port)
