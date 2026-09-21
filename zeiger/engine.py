"""The engine: load a checkpoint once, answer typed questions about a page."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .encoding import KINDS, Encoded, Question, TokenCache, batches, collate, encode
from .model import WINDOW, Zeiger, load

Answer = dict[str, Any]


def _bucket(n: int) -> str:
    return "2" if n <= 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine:
    """Answers questions about a page.

        engine = Engine("models/zeiger-0.6b")
        answers = engine.decide(state, {"q": {"type": "choice", "instructions": ..., "criteria": {...}}})
    """

    def __init__(self, checkpoint: str | Path, *, device: str | torch.device = "auto", max_len: int | None = None,
                 threads: int | None = None, gpu_memory_gb: float = 0.0, **options: Any) -> None:
        self.device = resolve_device(device)
        if threads and self.device.type == "cpu":
            torch.set_num_threads(threads)
        if gpu_memory_gb and self.device.type == "cuda":
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            total = torch.cuda.get_device_properties(index).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_memory_gb * 2**30 / total), index)
        self._configure(*load(checkpoint, max_len), **options)

    @classmethod
    def from_parts(cls, model: Zeiger, tokenizer: Any, config: Mapping[str, Any], *,
                   device: str | torch.device = "cpu", **options: Any) -> "Engine":
        """Wrap an already-loaded model - for tests, or when the weights come from somewhere else."""
        engine = cls.__new__(cls)
        engine.device = resolve_device(device)
        engine._configure(model, tokenizer, dict(config), **options)
        return engine

    def _configure(self, model: Zeiger, tokenizer: Any, config: dict, *, chunk_tokens: int | None = None,
                   token_budget: int = 16384, dtype: torch.dtype | None = None, warmup: bool | list[int] = False,
                   cache_tokens: int = 100_000, release_after: float = 0.0) -> None:
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.model.to(self.device).eval()
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.max_len = int(config.get("max_len") or WINDOW)
        self.chunk_tokens = config.get("chunk_tokens", 0) if chunk_tokens is None else chunk_tokens
        self.token_budget = token_budget
        self.temperatures = dict(config.get("temperature_by_options", {}))
        self.cache = TokenCache(cache_tokens) if cache_tokens else None
        self.release_after = release_after
        self._resident = True
        self._last_used = time.monotonic()
        self.stats = {"questions": 0, "batches": 0, "ms": 0.0}
        if warmup:
            self.warmup(warmup if isinstance(warmup, list) else None)

    def decide(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Answer]:
        """questions: {id: {"type", "instructions", "criteria"}} -> {id: answer}.

        A question whose options do not fit the window is reported rather than guessed at.
        """
        started = time.perf_counter()
        self._ensure_resident()
        answers: dict[str, Answer] = {}
        pending: list[tuple[str, Encoded]] = []

        for qid, raw in questions.items():
            question = Question(state=state, kind=raw.get("type", "choice"),
                                instructions=raw.get("instructions", ""), criteria=raw.get("criteria"))
            item = encode(self.tokenizer, question, self.max_len, self.chunk_tokens, self.cache)
            if item is None:
                answers[qid] = {"type": question.kind, "error": "options do not fit the window"}
            else:
                item.slot = len(pending)
                pending.append((qid, item))

        for (qid, item), logits in zip(pending, self._logits([item for _, item in pending])):
            answers[qid] = self._answer(item, logits)

        self.stats["questions"] += len(pending)
        self.stats["ms"] += (time.perf_counter() - started) * 1000
        self._last_used = time.monotonic()
        return answers

    def warmup(self, option_counts: list[int] | None = None) -> float:
        """Answer throwaway questions so kernels and buffers are ready before the first real one.

        The first call on a cold device pays for kernel selection and allocator growth; this moves that cost to
        start-up. Returns the seconds spent.
        """
        started = time.perf_counter()
        self._ensure_resident()
        for count in option_counts or [8, 64, 512]:
            criteria = {f"w{i}": f"warmup option {i}" for i in range(count)}
            self.decide({"instruction": "warmup"}, {"w": {"type": "choice", "instructions": "warmup", "criteria": criteria}})
        self.stats.update(questions=0, batches=0, ms=0.0)
        return round(time.perf_counter() - started, 3)

    def release(self) -> None:
        """Move the weights off the accelerator, keeping the process alive. The next call reloads them."""
        if not self._resident or self.device.type == "cpu":
            return
        self.model.to("cpu")
        torch.cuda.empty_cache()
        self._resident = False

    def _ensure_resident(self) -> None:
        if not self._resident:
            self.model.to(self.device)
            self._resident = True
        elif self.release_after and time.monotonic() - self._last_used > self.release_after:
            self.release()
            self.model.to(self.device)
            self._resident = True

    def info(self) -> dict[str, Any]:
        cache = {"entries": len(self.cache), "hits": self.cache.hits, "misses": self.cache.misses} if self.cache else None
        return {"arch": self.config.get("arch"), "encoder": self.config.get("encoder"), "max_len": self.max_len,
                "chunk_tokens": self.chunk_tokens, "device": str(self.device), "resident": self._resident,
                "dtype": str(self.dtype).removeprefix("torch."), "temperatures": self.temperatures,
                "token_cache": cache}

    @torch.no_grad()
    def _logits(self, items: list[Encoded]) -> list[np.ndarray]:
        out: list[np.ndarray | None] = [None] * len(items)
        for batch in batches(items, self.token_budget):
            inputs = collate(batch, self.tokenizer.pad_token_id)
            with self._autocast():
                logits, _ = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            logits = logits.float().cpu().numpy()
            for row, item in enumerate(batch):
                out[item.slot] = logits[row, : len(item.markers)]
            self.stats["batches"] += 1
        return out  # type: ignore[return-value]

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=self.dtype)
        return torch.autocast("cpu", dtype=self.dtype, enabled=self.dtype is torch.bfloat16)

    def _answer(self, item: Encoded, logits: np.ndarray) -> Answer:
        kind = KINDS[item.qtype]
        temperature = float(self.temperatures.get(f"{kind}:{_bucket(len(item.keys))}", 1.0))
        scaled = logits / max(temperature, 1e-3)
        probabilities = np.exp(scaled - scaled.max())
        probabilities /= probabilities.sum()
        by_key = {key: round(float(p), 5) for key, p in zip(item.keys, probabilities)}

        if kind == "noul":
            return {"type": "noul", "noul": by_key.get("true", 0.0)}
        best = max(by_key, key=by_key.get)
        if kind == "score":
            return {"type": "score", "score": item.keys.index(best), "confidence": by_key[best],
                    "probabilities": by_key}
        return {"type": "choice", "choice": best, "confidence": by_key[best], "probabilities": by_key}
