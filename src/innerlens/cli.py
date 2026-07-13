"""innerlens CLI: `innerlens serve` (OpenAI-compatible API) and
`innerlens demo` (the offline introspection demo)."""
from __future__ import annotations

import argparse
from typing import List, Optional

from innerlens import __version__


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="innerlens",
                                description="See what your open model is really thinking.")
    p.add_argument("--version", action="version", version=f"innerlens {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the OpenAI-compatible server")
    s.add_argument("--model", default=None, help="HF model id (default: Qwen/Qwen3.5-4B)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)

    d = sub.add_parser("demo", help="run the offline introspection demo")
    d.add_argument("--model", default="Qwen/Qwen3.5-4B")

    args = p.parse_args(argv)
    if args.cmd == "serve":
        from innerlens.server import serve
        serve(args.model, args.host, args.port)
        return 0
    if args.cmd == "demo":
        from innerlens.demo import run_demo
        return run_demo(args.model)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
