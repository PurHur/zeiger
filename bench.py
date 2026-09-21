#!/usr/bin/env python
"""What this box can do: ms per question against option count, device and batch size.

  python bench.py --model models/zeiger-0.6b --devices cuda,cpu

ROCm vs Vulkan, since the question always comes up: PyTorch has no Vulkan inference backend for a model like this
one (a Qwen3 backbone with a custom marker head), and llama.cpp - which does have a fast Vulkan path on this GPU -
cannot run that head at all. So on AMD the choice is ROCm or the CPU; a Vulkan path would mean reimplementing the
model in a Vulkan runtime, and this benchmark measures what the two real options cost.
"""
import argparse, json, random, statistics, sys, time
sys.path.insert(0, ".")
from zeiger import Engine

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/zeiger-0.6b")
ap.add_argument("--devices", default="cuda,cpu")
ap.add_argument("--options", default="10,60,150,400,1500")
ap.add_argument("--batches", default="1,8,32")
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--threads", type=int, default=16)
args = ap.parse_args()

def question(n, seed=0):
    r = random.Random(seed)
    crit = {f"e{i}": f'{r.choice(["link","button","input","span"])} "{r.choice(["Sign in","Basket","Search","Next page","Contact us","Add to cart"])} {i}" in {r.choice(["header","main","nav","footer"])}' for i in range(n - 1)}
    crit["none"] = "none of the listed elements fits this step"
    return {"type": "choice", "instructions": "Which page element does the instruction refer to?", "criteria": crit}

STATE = {"instruction": 'Click the "Sign in" button', "page": {"title": "Shop", "url": "https://shop.test/"}}
rows = []
for dev in [d for d in args.devices.split(",") if d]:
    try:
        e = Engine(args.model, device=dev, threads=args.threads)
    except Exception as exc:                      # a box without ROCm/CUDA just skips that row
        print(f"{dev}: unavailable ({str(exc)[:80]})", flush=True)
        continue
    print(f"\n{dev}: {json.dumps(e.info())}", flush=True)
    print(f"{'options':>8} {'batch':>6} {'ms/question':>12} {'questions/s':>12}")
    for n in [int(x) for x in args.options.split(",")]:
        for bs in [int(x) for x in args.batches.split(",")]:
            qs = {f"q{i}": question(n, i) for i in range(bs)}
            e.decide(STATE, {"q0": question(n)})   # warm up the shapes
            ts = []
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                out = e.decide(STATE, qs)
                ts.append((time.perf_counter() - t0) * 1000)
            if any("error" in a for a in out.values()):
                print(f"{n:>8} {bs:>6} {'does not fit':>12}")
                continue
            ms = statistics.median(ts) / bs
            rows.append({"device": dev, "options": n, "batch": bs, "ms_per_question": round(ms, 1)})
            print(f"{n:>8} {bs:>6} {ms:>12.1f} {1000/ms:>12.1f}", flush=True)
json.dump(rows, open("bench-results.json", "w"), indent=1)
print("\nwritten to bench-results.json", flush=True)
