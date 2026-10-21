"""The fixed benchmark data: lazy reading, the seeded training stream and the data-transform registry.

`bench/data/train.jsonl` is already shuffled (seed 0) by scripts/build_bench.py. A run with seed 0 reads it in file
order; any other seed reads a seeded permutation of it. Transforms get their own RNG per example
(seeded by run seed + example position), so what example i looks like never depends on batching or timing.

Extension point: `TRANSFORMS` (name -> class with `__init__(**cfg)` and `__call__(ex, rng, train) -> ex | None`).
List transforms in variant.json "transforms": [{"kind": name, ...cfg}]. A transform returning None drops the
example. Transforms run on the raw example dict (state / question / label / meta) before encoding; they may also
set `ex["_order"]` (option permutation) which the encoding honours. Eval examples pass train=False; a transform
that only augments training must return `ex` unchanged then (contract: branches never change the eval data).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterator

from .encoding import ENCODINGS, Item, option_keys
from .registry import Registry

SETS = ("short", "long", "general", "structured", "logic")
DEFAULT_LAB = Path("/home/ai/projects/zeiger-lab")


def lab_root() -> Path:
    """The main checkout (shared by all worktrees): runs/, bench/data/ and .trash/ live there."""
    env = os.environ.get("ZLAB_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent.parent
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=here,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        if common:
            return Path(common).parent
    except (OSError, subprocess.SubprocessError):
        pass
    return DEFAULT_LAB


def data_dir() -> Path:
    return Path(os.environ.get("ZLAB_DATA") or lab_root() / "bench" / "data")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_data(sha_file: Path, directory: Path | None = None) -> None:
    """Every file pinned in bench/data.sha256 must exist with that hash (the benchmark is fixed)."""
    directory = directory or data_dir()
    for line in Path(sha_file).read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = directory / Path(name.strip().lstrip("*")).name
        if not path.exists():
            raise FileNotFoundError(f"benchmark file missing: {path} (run scripts/build_bench.py)")
        got = sha256_file(path)
        if got != digest:
            raise ValueError(f"benchmark file {path} has sha256 {got}, pinned {digest}: the data changed")


class JsonlIndex:
    """Random access to a big jsonl without holding it in memory (train.jsonl holds whole pages)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offsets: list[int] = []
        pos = 0
        with open(self.path, "rb") as f:
            for line in f:
                if line.strip():
                    self.offsets.append(pos)
                pos += len(line)
        self._f = open(self.path, "rb")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, i: int) -> dict:
        self._f.seek(self.offsets[i])
        return json.loads(self._f.readline())


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def category(ex: dict) -> str:
    """short | long | general | structured | logic (the benchmark's five slices)."""
    if ex.get("_cat"):
        return ex["_cat"]
    src = ex.get("source", "")
    if src == "structured-logic":
        return "logic"
    if src.startswith("structured-"):
        return "structured"
    if src.startswith("long-"):
        return "long"
    state = ex.get("state")
    if isinstance(state, dict) and "page" in state:
        return "short"
    return "general"


# ------------------------------------------------------------------------------------------------- transforms

TRANSFORMS: Registry[type] = Registry("transform")


class Transform:
    def __init__(self, **cfg: Any) -> None:
        self.cfg = cfg

    def __call__(self, ex: dict, rng: random.Random, train: bool) -> dict | None:
        return ex


@TRANSFORMS.register("shuffle_options")
class ShuffleOptions(Transform):
    """Round 17's option permutation: choice questions get their options shuffled with probability p during
    training (the label follows). Ordinal (score) and yes/no (noul) questions are never shuffled."""

    def __init__(self, p: float = 0.5, **cfg: Any) -> None:
        super().__init__(p=p, **cfg)
        self.p = p

    def __call__(self, ex, rng, train):
        if not train or ex["question"].get("type", "choice") != "choice":
            return ex
        if rng.random() < self.p:
            order = list(range(len(option_keys(ex["question"]))))
            rng.shuffle(order)
            ex = {**ex, "_order": order}
        return ex


def build_transforms(specs: list[dict] | None) -> list[Transform]:
    out = []
    for spec in specs or []:
        spec = dict(spec)
        out.append(TRANSFORMS.get(spec.pop("kind"))(**spec))
    return out


def apply_transforms(ex: dict, transforms: list[Transform], rng: random.Random, train: bool) -> dict | None:
    for t in transforms:
        ex = t(ex, rng, train)
        if ex is None:
            return None
    return ex


def encode(tok, ex: dict, encoding: dict, max_len: int) -> Item | None:
    fn = ENCODINGS.get(encoding.get("kind", "chunked-markers"))
    item = fn(tok, ex, max_len, int(encoding.get("chunk_tokens", 0)), ex.get("_order"))
    if item is not None:
        item.cat = category(ex)
    return item


# --------------------------------------------------------------------------------------------- training stream

class TrainStream:
    """Deterministic stream of encoded training items: position p -> example order[p] -> transforms -> encoding.

    `skipped` counts examples whose options did not fit train_max_len (never trained on a truncated option list).
    """

    def __init__(self, tok, path: Path, seed: int, encoding: dict, transforms: list[Transform], max_len: int) -> None:
        self.tok, self.seed, self.encoding, self.transforms, self.max_len = tok, seed, encoding, transforms, max_len
        self.rows = JsonlIndex(path)
        self.order = list(range(len(self.rows)))
        if seed != 0:
            random.Random(seed).shuffle(self.order)
        self.pos = 0
        self.skipped = 0
        self.epoch = 0

    def __iter__(self) -> Iterator[Item]:
        return self

    def __next__(self) -> Item:
        while True:
            if self.pos >= len(self.order):
                self.pos = 0
                self.epoch += 1
            p = self.pos
            self.pos += 1
            ex = self.rows[self.order[p]]
            rng = random.Random(f"{self.seed}:{self.epoch}:{p}")
            ex = apply_transforms(ex, self.transforms, rng, train=True)
            item = encode(self.tok, ex, self.encoding, self.max_len) if ex is not None else None
            if item is None:
                self.skipped += 1
                continue
            return item

    def next_step(self, tokens_per_step: int) -> list[Item]:
        """Items for one optimizer step: filled until their real tokens reach tokens_per_step (laya's length-aware steps)."""
        group: list[Item] = []
        total = 0
        while total < tokens_per_step:
            item = next(self)
            group.append(item)
            total += item.n_tokens
        return group


def load_eval_sets(limit_per_set: int | None = None, directory: Path | None = None) -> dict[str, list[dict]]:
    directory = directory or data_dir()
    out = {}
    for name in SETS:
        path = directory / f"eval_{name}.jsonl"
        if path.exists():
            rows = read_jsonl(path, limit_per_set)
            for r in rows:
                r["_cat"] = name
            out[name] = rows
    return out
