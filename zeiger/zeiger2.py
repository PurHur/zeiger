"""Zeiger2 checkpoints (arch "zlab-*", e.g. php-ai/zeiger2-3.5-0.8b): the same decide() as Engine, served by the
model code of the zeiger-lab training pipeline (the top-level zlab/ package of this repository).

Engine(...) returns a Zeiger2Engine by itself when the checkpoint's config says "zlab-*"; callers do not change.
Needs transformers >= 5 (Qwen3.5 backbone). Every question is answered twice, with the options in the shown and in
the reversed order, and the probabilities are averaged per option (two-order, part of the trained configuration);
two_order=False halves the latency.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .engine import Answer, Engine, _cpu, resolve_device

ROOT = Path(__file__).resolve().parent.parent  # the repository: zlab/ lives next to zeiger/
KINDS = ("choice", "noul", "score")


def is_zeiger2(config: Mapping[str, Any] | None) -> bool:
    return bool(config) and str(config.get("arch", "")).startswith("zlab-")


class Zeiger2Engine(Engine):
    """Engine for Zeiger2 checkpoints. Same constructor options as Engine where they apply (device, threads,
    gpu_memory_gb, token_budget, warmup, release_after); chunk size, window and encoding come from the checkpoint."""

    def __init__(self, checkpoint: str | Path, *, device: str | torch.device = "auto", max_len: int | None = None,
                 threads: int | None = None, gpu_memory_gb: float = 0.0, token_budget: int = 32768,
                 warmup: bool | list[int] = False, release_after: float = 0.0, two_order: bool = True,
                 **_ignored: Any) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from zlab import evaluate as ev
        from zlab.data import build_transforms
        from zlab.encoding import option_keys
        from zlab.model import build_model

        self.device = resolve_device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(threads or _cpu()[0])
        if gpu_memory_gb and self.device.type == "cuda":
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            total = torch.cuda.get_device_properties(index).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_memory_gb * 2**30 / total), index)
        path = Path(checkpoint)
        self.config = json.loads((path / "config.json").read_text())
        variant = self.config["zlab_variant"]
        self._ev, self._option_keys = ev, option_keys
        self.model, self.tokenizer, _ = build_model(variant, str(path))
        self.model.to(self.device).eval()
        self.encoding = variant.get("encoding") or {"kind": "chunked-markers", "chunk_tokens": 2048}
        self.transforms = build_transforms(variant.get("transforms"))
        self.hooks = ev.build_eval_hooks(variant.get("eval_hooks")) if two_order else []
        self.max_len = int(max_len or self.config.get("max_len") or 131072)
        self.chunk_tokens = int(self.encoding.get("chunk_tokens", 0))
        self.token_budget = token_budget
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.temperatures: dict = {}
        self.cache = None
        self.release_after = release_after
        self._resident = True
        self._last_used = time.monotonic()
        self.stats = {"questions": 0, "batches": 0, "ms": 0.0}
        if warmup:
            self.warmup(warmup if isinstance(warmup, list) else None)

    def _predict(self, rows: list[dict], order_fn=None):
        return self._ev.predict(self.model, self.tokenizer, rows, self.encoding, self.max_len, self.token_budget,
                                self.device, self.transforms, order_fn=order_fn)

    @torch.no_grad()
    def decide(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Answer]:
        started = time.perf_counter()
        self._ensure_resident()
        rows = []
        for qid, raw in questions.items():
            kind = raw.get("type", "choice")
            if kind not in KINDS:
                raise ValueError(f"question {qid}: type must be one of {KINDS}")
            instructions = raw.get("instructions", "")
            criteria = raw.get("criteria")
            if kind == "noul":
                criteria = criteria or {}
            question = {"type": kind, "criteria": criteria if isinstance(criteria, dict) else list(criteria or ()),
                        "instructions": instructions if isinstance(instructions, str) else json.dumps(instructions, ensure_ascii=False)}
            keys = self._option_keys(question)
            # the encoder needs a valid label to build a row; inference never reads it
            rows.append({"id": qid, "source": "engine", "state": state, "question": question, "label": keys[0] if keys else "none"})
        recs, _ = self._predict(rows)
        for hook in self.hooks:
            recs, _, _ = hook("engine", rows, recs, lambda sub, order_fn=None: self._predict(sub, order_fn=order_fn))

        answers: dict[str, Answer] = {}
        for (qid, raw), rec in zip(questions.items(), recs):
            kind = raw.get("type", "choice")
            if rec["probs"] is None:
                answers[qid] = {"type": kind, "error": "options do not fit the window"}
                continue
            by_key = {key: round(float(p), 5) for key, p in zip(rec["keys"], rec["probs"])}
            if kind == "noul":
                answers[qid] = {"type": "noul", "noul": by_key.get("true", 0.0)}
                continue
            keys = list(raw["criteria"]) if isinstance(raw.get("criteria"), dict) else [str(i) for i in range(len(raw.get("criteria") or ()))]
            by_key = {key: by_key.get(key, 0.0) for key in keys}  # the order the caller gave
            best = max(by_key, key=by_key.get)
            if kind == "score":
                answers[qid] = {"type": "score", "score": keys.index(best), "confidence": by_key[best], "probabilities": by_key}
            else:
                answers[qid] = {"type": "choice", "choice": best, "confidence": by_key[best], "probabilities": by_key}

        self.stats["questions"] += len(rows)
        self.stats["ms"] += (time.perf_counter() - started) * 1000
        self._last_used = time.monotonic()
        return answers

    def info(self) -> dict[str, Any]:
        return {**super().info(), "two_order": bool(self.hooks), "token_cache": None}
