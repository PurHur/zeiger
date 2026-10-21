"""Option-order eval passes (decisions `order-free` / two-order; Juspay Xor, lev).

Xor and lev read every choice and binary question twice, in the shown and in the reversed option order, and average
(Xor: probabilities for binary questions, mean logprobs then softmax for >= 3 options; lev: both; score questions are
not averaged). Two eval hooks (zlab.evaluate.EVAL_HOOKS) on top of the plain pass of each eval set:

- "reversed_order" (diagnostic, arch/order-free): one more pass with the options reversed; the set's metrics stay
  those of the shown order, and the set dict gets
      order_flip_rate  fraction of reversible questions whose top-1 option changes when the order is reversed
      order_tv         mean total-variation distance between the two distributions (Strands reports ~0.016)
      order_n          questions compared (both passes fit), order_ms_per_q  cost of the extra pass
  The extra pass is a measurement, not model cost: its seconds do not count in ms_per_q.
- "two_order" (arch/two-order): the same reversed pass, then each question's probabilities are the average of the
  two orders (combine "prob", default, or "logprob" = softmax of the mean log-probabilities, Xor's >= 3-option rule)
  and every metric of the set is computed on the averaged distribution. It is the model's real cost (2x forwards),
  so ms_per_q includes it. order_flip_rate / order_tv / order_n are reported too.

Reversible questions: choice and noul with >= 2 options (types cfg); score levels are ordinal and stay in order (lev).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from zlab.encoding import option_keys
from zlab.evaluate import EVAL_HOOKS, EvalHook


def reversible(ex: dict, types: tuple[str, ...]) -> bool:
    q = ex["question"]
    return q.get("type", "choice") in types and len(option_keys(q)) >= 2


def reversed_order(ex: dict) -> list[int]:
    return list(range(len(option_keys(ex["question"]))))[::-1]


def align(rec: dict, keys: list[str]) -> np.ndarray:
    """rec's probabilities in the order of `keys`."""
    by_key = dict(zip(rec["keys"], rec["probs"]))
    return np.array([by_key[k] for k in keys], dtype=np.float64)


def combine(p1: np.ndarray, p2: np.ndarray, how: str) -> np.ndarray:
    if how == "logprob":
        z = 0.5 * (np.log(np.clip(p1, 1e-30, None)) + np.log(np.clip(p2, 1e-30, None)))
        z = np.exp(z - z.max())
        return z / z.sum()
    return 0.5 * (p1 + p2)


@EVAL_HOOKS.register("reversed_order")
class ReversedOrder(EvalHook):
    adds_cost = False
    averages = False

    def __init__(self, types: list[str] | tuple[str, ...] = ("choice", "noul"), combine: str = "prob", **cfg: Any) -> None:
        super().__init__(types=list(types), combine=combine, **cfg)
        if combine not in ("prob", "logprob"):
            raise ValueError(f"combine must be prob or logprob, not {combine!r}")
        self.types, self.how = tuple(types), combine

    def __call__(self, name, rows, recs, predict_fn):
        idx = [i for i, ex in enumerate(rows) if reversible(ex, self.types)]
        if not idx:
            return recs, {"order_flip_rate": None, "order_tv": None, "order_n": 0}, 0.0
        rev, secs = predict_fn([rows[i] for i in idx], reversed_order)
        flips, tvs = [], []
        out = list(recs)
        for i, r2 in zip(idx, rev):
            r1 = recs[i]
            if r1["probs"] is None or r2["probs"] is None:
                if self.averages and r1["probs"] is None and r2["probs"] is not None:
                    out[i] = {**r1, "probs": r2["probs"], "keys": r2["keys"]}
                continue
            keys = list(r1["keys"])
            p1, p2 = np.asarray(r1["probs"], dtype=np.float64), align(r2, keys)
            flips.append(int(np.argmax(p1)) != int(np.argmax(p2)))
            tvs.append(0.5 * float(np.abs(p1 - p2).sum()))
            if self.averages:
                out[i] = {**r1, "probs": combine(p1, p2, self.how), "keys": keys}
        n = len(flips)
        extra = {"order_flip_rate": round(float(np.mean(flips)), 4) if n else None,
                 "order_tv": round(float(np.mean(tvs)), 4) if n else None, "order_n": n,
                 "order_ms_per_q": round(secs * 1000 / max(1, len(idx)), 1)}
        return out, extra, secs


@EVAL_HOOKS.register("two_order")
class TwoOrder(ReversedOrder):
    adds_cost = True
    averages = True
