"""Turning a page and its options into tokens."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

MARKER = "<|object_ref_end|>"
QTYPES = {"choice": 0, "score": 1, "noul": 2}
KINDS = ("choice", "score", "noul")

OPTION_TOKENS = 200
STATE_TOKENS = 1536
HEAD_TOKENS = 512
PAD_MULTIPLE = 256

Chunk = tuple[list[int], list[int]]


@dataclass(slots=True)
class Question:
    """A typed question: what the agent is doing, what is being asked, and the options."""

    state: Any
    kind: str = "choice"
    instructions: str = ""
    criteria: dict[str, str] | Sequence[str] | None = None

    @property
    def keys(self) -> list[str]:
        if isinstance(self.criteria, dict):
            return list(self.criteria)
        return [str(i) for i in range(len(self.criteria or ()))]

    @property
    def qtype(self) -> int:
        return QTYPES.get(self.kind, 0)

    def options(self) -> list[str]:
        if self.kind == "choice":
            return [k if not v else f"{k}: {v}" for k, v in (self.criteria or {}).items()]
        if self.kind == "score":
            return [f"level {i}: {c}" for i, c in enumerate(self.criteria or ())]
        crit = self.criteria if isinstance(self.criteria, dict) else {}
        return [f"false: {crit.get('false') or 'no, the statement does not hold'}",
                f"true: {crit.get('true') or 'yes, the statement holds'}"]


@dataclass(slots=True)
class Encoded:
    """One question, tokenised: the chunks it spans and where its option markers sit."""

    chunks: list[Chunk]
    keys: list[str]
    qtype: int
    slot: int = 0
    markers: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.markers = [m for _, ms in self.chunks for m in ms]

    @property
    def longest_chunk(self) -> int:
        return max((len(ids) for ids, _ in self.chunks), default=0)

    @property
    def cost(self) -> int:
        return sum(-(-len(ids) // PAD_MULTIPLE) * PAD_MULTIPLE for ids, _ in self.chunks)


def serialise(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def encode(tok, question: Question, max_len: int, chunk_tokens: int = 0) -> Encoded | None:
    """Tokenise a question, or return None when its options do not fit `max_len`.

    A question that does not fit is never answered on a truncated option list: the caller reports it instead.
    """
    chunks = _chunks(tok, question, max_len, chunk_tokens)
    if sum(len(ms) for _, ms in chunks) != len(question.keys):
        return None
    return Encoded(chunks=chunks, keys=question.keys, qtype=question.qtype)


def _tokenise(tok, text: str) -> list[int]:
    clean = text.replace(MARKER, " ").encode("utf-8", "ignore").decode("utf-8")
    return tok(clean, add_special_tokens=False)["input_ids"]


def _sequence(tok, question: Question, max_len: int) -> list[Chunk]:
    """One sequence: state, question, then every option followed by its marker."""
    marker, eos = tok.convert_tokens_to_ids(MARKER), tok.eos_token_id or tok.pad_token_id
    head = _tokenise(tok, f"\n{question.kind} question: {question.instructions}\n")[:HEAD_TOKENS]
    options = [_tokenise(tok, " " + text)[:OPTION_TOKENS] for text in question.options()]
    room = max_len - (len(head) + sum(len(o) + 1 for o in options) + 1)
    ids = _tokenise(tok, serialise(question.state))[: max(64, min(STATE_TOKENS, room))] + head
    markers = []
    for option in options:
        ids.extend(option)
        markers.append(len(ids))
        ids.append(marker)
    ids.append(eos)
    if len(ids) > max_len:
        return [(ids[:max_len], [m for m in markers if m < max_len])]
    return [(ids, markers)]


def _chunks(tok, question: Question, max_len: int, chunk_tokens: int) -> list[Chunk]:
    """Chunked-prefix encoding: every chunk repeats the state and question, then carries a run of options.

    Attention cost is linear in page size, and a page is not limited by the backbone's window. A question that
    fits one chunk is encoded as a single sequence, byte for byte.
    """
    if not chunk_tokens:
        return _sequence(tok, question, max_len)
    marker, eos = tok.convert_tokens_to_ids(MARKER), tok.eos_token_id or tok.pad_token_id
    head = _tokenise(tok, f"\n{question.kind} question: {question.instructions}\n")[:HEAD_TOKENS]
    options = [_tokenise(tok, " " + text)[:OPTION_TOKENS] for text in question.options()]
    state = _tokenise(tok, serialise(question.state))
    if len(state[:STATE_TOKENS]) + len(head) + sum(len(o) + 1 for o in options) + 1 <= chunk_tokens:
        return _sequence(tok, question, max(max_len, chunk_tokens))

    prefix = state[: min(STATE_TOKENS, max(64, chunk_tokens // 2 - len(head)))] + head
    room = max(OPTION_TOKENS + 1, chunk_tokens - len(prefix) - 1)
    chunks: list[Chunk] = []
    ids, markers, used, total = list(prefix), [], 0, 0
    for option in options:
        if used and used + len(option) + 1 > room:
            chunks.append((ids + [eos], markers))
            total += len(ids) + 1
            ids, markers, used = list(prefix), [], 0
        if total + len(ids) + len(option) + 2 > max_len:
            break
        ids.extend(option)
        markers.append(len(ids))
        ids.append(marker)
        used += len(option) + 1
    if markers:
        chunks.append((ids + [eos], markers))
    return chunks


def collate(items: Sequence[Encoded], pad_id: int) -> dict[str, torch.Tensor]:
    """Chunks of several questions into one padded batch, plus the map from options to marker states."""
    chunks = [(i, ids, markers) for i, item in enumerate(items) for ids, markers in item.chunks]
    width = -(-max(len(ids) for _, ids, _ in chunks) // PAD_MULTIPLE) * PAD_MULTIPLE
    per_chunk = max(len(ms) for _, _, ms in chunks)
    n_options = max(len(item.markers) for item in items)

    input_ids = torch.full((len(chunks), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(chunks), width), dtype=torch.long)
    marker_pos = torch.zeros((len(chunks), per_chunk), dtype=torch.long)
    index = torch.zeros((len(items), n_options), dtype=torch.long)
    mask = torch.zeros((len(items), n_options), dtype=torch.bool)

    filled = [0] * len(items)
    for row, (i, ids, markers) in enumerate(chunks):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        attention[row, : len(ids)] = 1
        if not markers:
            continue
        marker_pos[row, : len(markers)] = torch.tensor(markers)
        start = filled[i]
        index[i, start : start + len(markers)] = row * per_chunk + torch.arange(len(markers))
        mask[i, start : start + len(markers)] = True
        filled[i] = start + len(markers)

    return {"input_ids": input_ids, "attention_mask": attention, "marker_pos": marker_pos,
            "marker_mask": mask, "q_index": index,
            "qtype": torch.tensor([item.qtype for item in items])}


def batches(items: Iterable[Encoded], budget: int) -> Iterable[list[Encoded]]:
    """Micro-batches under a padded-token budget, grouped by chunk length.

    Every row of a batch is padded to the longest one in it, so a short question batched with a full page would
    pay for the page. Padding is masked either way: this changes speed, never an answer.
    """
    items = list(items)
    bins = ([i for i in items if i.longest_chunk <= 512],
            [i for i in items if 512 < i.longest_chunk <= 1024],
            [i for i in items if i.longest_chunk > 1024])
    for group in bins:
        batch, used = [], 0
        for item in sorted(group, key=lambda i: i.longest_chunk):
            if batch and used + item.cost > budget:
                yield batch
                batch, used = [], 0
            batch.append(item)
            used += item.cost
        if batch:
            yield batch
