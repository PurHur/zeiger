"""Evaluation on the frozen benchmark sets (bench/data/eval_*.jsonl).

Per set: n, top1, none_f1, ece, brier, ms_per_q — defined as in laya-browser's metrics.py:
- a question whose options do not fit eval_max_len counts as WRONG (what production does), with confidence 1.0;
- probabilities are a plain softmax (temperature 1): per-bucket temperature fitting is a post-hoc step and would
  hide calibration differences between variants;
- none_f1 is None for a set without 'none' labels and without 'none' predictions;
- ece: 15 equal-width confidence bins (top-1 confidence vs correctness); brier: mean over questions of
  sum_k (p_k - onehot_k)^2 (2.0 for a question that did not fit).

score = 0.4 * short.top1 + 0.4 * long.top1 + 0.2 * general.top1.

Extension point `EVAL_HOOKS` (variant.json "eval_hooks": [{"kind": name, ...cfg}]): a hook runs after the plain pass
of every eval set, may run more passes through `predict_fn(rows, order_fn)` (order_fn(ex) -> option order or None),
may replace the per-question probabilities (e.g. averaged over option orders) and returns extra metrics that are
added to the set's dict next to n/top1/none_f1/ece/brier/ms_per_q (contract-compatible: readers ignore extra keys).
A hook whose passes are part of the model's real cost sets `adds_cost = True`, so its seconds count in ms_per_q.

    python -m zlab.evaluate --model /home/ai/projects/zeiger/models/zeiger-0.6b --limit 50
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

import numpy as np
import torch

from .data import SETS, apply_transforms, build_transforms, encode, load_eval_sets
from .encoding import collate, token_batches
from .registry import Registry

SCORE_WEIGHTS = {"short": 0.4, "long": 0.4, "general": 0.2}


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def predict(model, tok, rows: list[dict], encoding: dict, max_len: int, token_budget: int, device: torch.device,
            transforms=None, order_fn=None) -> tuple[list[dict], float]:
    """-> ([{ex, probs (np) | None, keys}], forward seconds). Items are re-ordered by batching; results are written
    back by slot, never zipped with the input order. order_fn(ex) -> option order (or None) presents the options of
    a question in another order (eval hooks); probs/keys come back in the presented order."""
    was_training = model.training
    model.eval()
    recs: list[dict] = [{"ex": ex, "probs": None, "keys": None} for ex in rows]
    items = []
    for slot, ex in enumerate(rows):
        ex2 = apply_transforms(ex, transforms, random.Random(f"eval:{slot}"), train=False) if transforms else ex
        if order_fn is not None and ex2 is not None:
            order = order_fn(ex2)
            if order is not None:
                ex2 = {**ex2, "_order": list(order)}
        it = encode(tok, ex2, encoding, max_len) if ex2 is not None else None
        if it is not None:
            it.extra["slot"] = slot
            items.append(it)
    t0 = time.perf_counter()
    for batch_items in token_batches(items, token_budget):
        b = to_device(collate(batch_items, tok.pad_token_id), device)
        with autocast(device):
            logits = model(b)
        probs = torch.softmax(logits.float(), -1).cpu().numpy()
        for i, it in enumerate(batch_items):
            recs[it.extra["slot"]].update(probs=probs[i, : it.n_markers], keys=it.keys)
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    if was_training:
        model.train()
    return recs, secs


def metrics(recs: list[dict], secs: float) -> dict:
    ok, conf, brier = [], [], []
    tp = fp = fn = 0
    for r in recs:
        ex = r["ex"]
        label = str(ex["label"])
        is_none = label == "none"
        if r["probs"] is None:
            ok.append(0)
            conf.append(1.0)
            brier.append(2.0)
            fn += int(is_none)
            continue
        p, keys = r["probs"], r["keys"]
        top = int(np.argmax(p))
        li = keys.index(label)
        ok.append(int(top == li))
        conf.append(float(p[top]))
        onehot = np.zeros_like(p)
        onehot[li] = 1.0
        brier.append(float(((p - onehot) ** 2).sum()))
        pred_none = keys[top] == "none"
        tp += int(pred_none and is_none)
        fp += int(pred_none and not is_none)
        fn += int((not pred_none) and is_none)
    n = len(recs)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"n": n, "top1": round(float(np.mean(ok)) if n else 0.0, 4),
            "none_f1": round(2 * prec * rec / max(1e-9, prec + rec), 4) if (tp + fn) else None,
            "ece": round(ece_score(np.array(conf), np.array(ok)), 4) if n else None,
            "brier": round(float(np.mean(brier)), 4) if n else None,
            "ms_per_q": round(secs * 1000 / max(1, n), 1)}


def breakdown(recs: list[dict]) -> dict:
    """top-1 per source/variant (written to eval_detail_<step>.json, not part of the contract files)."""
    groups = defaultdict(list)
    for r in recs:
        ex = r["ex"]
        okv = 0
        if r["probs"] is not None:
            okv = int(r["keys"][int(np.argmax(r["probs"]))] == str(ex["label"]))
        groups[f"{ex.get('source')}/{(ex.get('meta') or {}).get('variant')}"].append(okv)
    return {k: {"n": len(v), "top1": round(float(np.mean(v)), 4)} for k, v in sorted(groups.items())}


def score(sets: dict) -> float | None:
    if not all(k in sets for k in SCORE_WEIGHTS):
        return None
    return round(sum(w * sets[k]["top1"] for k, w in SCORE_WEIGHTS.items()), 4)


EVAL_HOOKS: Registry[type] = Registry("eval hook")


class EvalHook:
    adds_cost = False

    def __init__(self, **cfg) -> None:
        self.cfg = cfg

    def __call__(self, name: str, rows: list[dict], recs: list[dict], predict_fn) -> tuple[list[dict], dict, float]:
        """-> (recs, extra metrics for the set's dict, seconds spent in extra passes)."""
        return recs, {}, 0.0


def build_eval_hooks(specs: list[dict] | None) -> list[EvalHook]:
    out = []
    for spec in specs or []:
        spec = dict(spec)
        out.append(EVAL_HOOKS.get(spec.pop("kind"))(**spec))
    return out


def evaluate(model, tok, eval_sets: dict[str, list[dict]], encoding: dict, max_len: int, token_budget: int,
             device: torch.device, transforms=None, on_set=None, hooks=None) -> tuple[dict, dict]:
    """-> ({set: metrics (+ hook metrics)}, {set: breakdown})."""
    out, detail = {}, {}

    def predict_fn(sub_rows, order_fn=None):
        return predict(model, tok, sub_rows, encoding, max_len, token_budget, device, transforms, order_fn=order_fn)

    for name in SETS:
        rows = eval_sets.get(name)
        if not rows:
            continue
        if on_set:
            on_set(name)
        recs, secs = predict(model, tok, rows, encoding, max_len, token_budget, device, transforms)
        extra = {}
        for hook in hooks or []:
            recs, more, hook_secs = hook(name, rows, recs, predict_fn)
            extra.update(more)
            if hook.adds_cost:
                secs += hook_secs
        out[name] = {**metrics(recs, secs), **extra}
        detail[name] = breakdown(recs)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Zeiger-format checkpoint directory")
    ap.add_argument("--variant", default="bench/variant.json")
    ap.add_argument("--budget", default="bench/budget.json")
    ap.add_argument("--limit", type=int, default=None, help="questions per set (default: all)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from .model import build_model
    from .safety import cap_gpu_memory

    variant = json.load(open(args.variant))
    budget = json.load(open(args.budget))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cap_gpu_memory(device, budget.get("gpu_mem_cap_gb", 0))
    model, tok, _ = build_model(variant, args.model)
    model.to(device)
    sets, _ = evaluate(model, tok, load_eval_sets(args.limit), variant.get("encoding") or {}, budget["eval_max_len"],
                       budget["eval_token_budget"], device, build_transforms(variant.get("transforms")),
                       hooks=build_eval_hooks(variant.get("eval_hooks")))
    print(json.dumps({"sets": sets, "score": score(sets)}, indent=1))


if __name__ == "__main__":
    main()
