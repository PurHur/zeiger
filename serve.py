#!/usr/bin/env python
"""Zeiger over HTTP: POST /decide {"state", "questions"} -> {"answers", "ms"}.

  python serve.py --model models/zeiger-0.6b [--port 8173] [--device auto|cuda|cpu]

The body shape matches what a browser agent already sends a decision model, so an existing caller only changes
its URL. GET / reports what is loaded. On AMD start it with
LD_PRELOAD=/opt/rocm-<ver>/lib/libhsa-runtime64.so.1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 (see README).
"""
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeiger import Engine

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("ZEIGER_MODEL", "models/zeiger-0.6b"))
ap.add_argument("--port", type=int, default=int(os.environ.get("ZEIGER_PORT", "8173")))
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--device", default=os.environ.get("ZEIGER_DEVICE", "auto"))
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--token-budget", type=int, default=16384)
ap.add_argument("--warmup", action="store_true", help="answer throwaway questions at start-up so the first real one is not the slow one")
ap.add_argument("--cache-tokens", type=int, default=int(os.environ.get("ZEIGER_CACHE_TOKENS", "100000")),
                help="remember tokenised option texts (0 disables); saves ~20 ms per repeated 300-option page")
ap.add_argument("--release-after", type=float, default=float(os.environ.get("ZEIGER_RELEASE_AFTER", "0")),
                help="seconds idle after which the weights leave the GPU, reloaded on the next request")
ap.add_argument("--gpu-memory-gb", type=float, default=0.0, help="cap the allocator on a shared GPU")
args = ap.parse_args()

engine = Engine(args.model, device=args.device, threads=args.threads or None,
                token_budget=args.token_budget, gpu_memory_gb=args.gpu_memory_gb,
                warmup=args.warmup or bool(int(os.environ.get("ZEIGER_WARMUP", "0"))),
                cache_tokens=args.cache_tokens, release_after=args.release_after)
LOCK = threading.Lock()   # one model, one GPU: serialise so batches stay whole
print(json.dumps({"ready": True, **engine.info(), "port": args.port}), flush=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send(200, {"ok": True, **engine.info(), "stats": engine.stats})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        except json.JSONDecodeError as exc:
            return self._send(400, {"error": f"bad JSON: {exc}"})
        qs = body.get("questions") or {}
        if not isinstance(qs, dict) or not qs:
            return self._send(400, {"error": "questions must be a non-empty object {id: {type, instructions, criteria}}"})
        t0 = time.perf_counter()
        try:
            with LOCK:
                answers = engine.decide(body.get("state") or {}, qs)
        except Exception as exc:  # a bad question must not take the server down
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        self._send(200, {"model": engine.info().get("arch"), "answers": answers, "ms": round((time.perf_counter() - t0) * 1000, 1)})

    def log_message(self, *a):
        pass


ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
