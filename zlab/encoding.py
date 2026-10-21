"""Question -> tokens, exactly as Zeiger is trained and served.

Ported from laya-browser `laya_browser/longctx.py` (build_sequence / build_chunks / collate_chunked) and checked
against the inference engine `zeiger/encoding.py` by tests/test_encoding.py (same ids, same marker positions):

    <state as JSON> \\n <type> question: <instructions> \\n  opt0 <M> opt1 <M> ... optK-1 <M> <eos>

Chunked-prefix encoding (chunk_tokens > 0): a question that does not fit one chunk is split into chunks that each
repeat the state + question prefix and carry a run of whole options. Options interact inside their chunk in the
backbone and across all chunks in the head.

Extension point: `ENCODINGS` (name -> function(tok, ex, max_len, chunk_tokens, order) -> Item | None). A branch
that needs another token layout (e.g. a decision-query token) registers its own encoding and names it in
variant.json "encoding.kind"; items may carry anything extra in `Item.extra` and `collate` passes per-chunk extras
listed in `Item.extra["chunk_fields"]` through.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

from .registry import Registry

MARKER = "<|object_ref_end|>"  # reserved in the Qwen tokenizer, unused in text
QTYPES = {"choice": 0, "score": 1, "noul": 2}
OPTION_TOKENS = 200
STATE_TOKENS = 1536
HEAD_TOKENS = 512
PAD_MULTIPLE = 256

Chunk = tuple[list[int], list[int]]


def serialize_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_options(q: dict) -> list[str]:
    """q = {"t": type, "ins": instructions, "crit": criteria} (laya's internal form)."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def option_keys(question: dict) -> list[str]:
    """The answer keys in option order: criteria keys (choice), level indices (score), false/true (noul)."""
    crit = question.get("criteria")
    t = question.get("type", "choice")
    if t == "noul":
        return ["false", "true"]
    if isinstance(crit, dict):
        return list(crit)
    return [str(i) for i in range(len(crit or ()))]


def marker_token(tok) -> str:
    """The option marker of this tokenizer: `tok.zlab_marker` (set by the backbone's load_tokenizer, a reserved token
    of that vocabulary, see zlab.backbones.MARKER_CANDIDATES), else Zeiger's <|object_ref_end|>."""
    return getattr(tok, "zlab_marker", None) or MARKER


def _tokenizer_fn(tok) -> Callable[[str], list[int]]:
    # crawled text may contain lone surrogates (half an emoji): the Rust tokenizer refuses them
    marker = marker_token(tok)
    return lambda s: tok(s.replace(marker, " ").encode("utf-8", "ignore").decode("utf-8"), add_special_tokens=False)["input_ids"]


def _eos(tok) -> int:
    return tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id


def build_sequence(tok, state, q, max_len, option_order=None) -> tuple[list[int], list[int], int]:
    """-> (ids, marker positions, prefix length). Options are never dropped silently: when they do not fit, fewer
    markers come back than options and the caller treats the question as not answerable."""
    mark = tok.convert_tokens_to_ids(marker_token(tok))
    enc = _tokenizer_fn(tok)
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    head = enc("\n%s question: %s\n" % (q["t"], str(q["ins"])))[:HEAD_TOKENS]
    opt_ids = [enc(" " + opts[i])[:OPTION_TOKENS] for i in order]
    need = len(head) + sum(len(o) + 1 for o in opt_ids) + 1
    st = enc(serialize_state(state))[: max(64, min(STATE_TOKENS, max_len - need))]
    ids = st + head
    prefix = len(ids)
    markers = []
    for o in opt_ids:
        ids.extend(o)
        markers.append(len(ids))
        ids.append(mark)
    ids.append(_eos(tok))
    if len(ids) > max_len:
        return ids[:max_len], [m for m in markers if m < max_len], min(prefix, max_len)
    return ids, markers, prefix


def build_chunks(tok, state, q, max_len, option_order=None, chunk_tokens=0) -> tuple[list[Chunk], list[int]]:
    """-> ([(ids, markers), ...], [prefix length per chunk]). chunk_tokens = 0, or a question that fits one chunk:
    exactly build_sequence(). max_len caps the TOTAL tokens over all chunks (memory), not one sequence."""
    if not chunk_tokens:
        ids, markers, prefix = build_sequence(tok, state, q, max_len, option_order)
        return [(ids, markers)], [prefix]
    mark = tok.convert_tokens_to_ids(marker_token(tok))
    eos = _eos(tok)
    enc = _tokenizer_fn(tok)
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    head = enc("\n%s question: %s\n" % (q["t"], str(q["ins"])))[:HEAD_TOKENS]
    opt_ids = [enc(" " + opts[i])[:OPTION_TOKENS] for i in order]
    need = len(head) + sum(len(o) + 1 for o in opt_ids) + 1
    full_state = enc(serialize_state(state))
    if len(full_state[:STATE_TOKENS]) + need <= chunk_tokens:  # fits one chunk: the plain sequence, bit for bit
        ids, markers, prefix = build_sequence(tok, state, q, max(max_len, chunk_tokens), option_order)
        return [(ids, markers)], [prefix]
    prefix = full_state[: min(STATE_TOKENS, max(64, chunk_tokens // 2 - len(head)))] + head
    room = max(OPTION_TOKENS + 1, chunk_tokens - len(prefix) - 1)
    chunks: list[Chunk] = []
    cur, cur_markers, used, total = list(prefix), [], 0, 0
    for o in opt_ids:
        if used and used + len(o) + 1 > room:
            chunks.append((cur + [eos], cur_markers))
            total += len(cur) + 1
            cur, cur_markers, used = list(prefix), [], 0
        if total + len(cur) + len(o) + 2 > max_len:
            break  # the page exceeds the memory cap: fewer markers than options -> the question is skipped / wrong
        cur.extend(o)
        cur_markers.append(len(cur))
        cur.append(mark)
        used += len(o) + 1
    if cur_markers:
        chunks.append((cur + [eos], cur_markers))
    return chunks, [len(prefix)] * len(chunks)


@dataclass
class Item:
    """One encoded question. `order[j]` = index (in the example's option list) of the j-th encoded option."""

    chunks: list[Chunk]
    prefix_lens: list[int]
    keys: list[str]
    qtype: int
    label: int
    ex_id: str = ""
    cat: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_markers(self) -> int:
        return sum(len(ms) for _, ms in self.chunks)

    @property
    def n_tokens(self) -> int:
        return sum(len(ids) for ids, _ in self.chunks)

    @property
    def longest_chunk(self) -> int:
        return max((len(ids) for ids, _ in self.chunks), default=0)

    @property
    def cost(self) -> int:
        """Padded tokens this item costs in a micro-batch."""
        return sum(-(-len(ids) // PAD_MULTIPLE) * PAD_MULTIPLE for ids, _ in self.chunks)


def to_internal(question: dict) -> dict:
    return {"t": question.get("type", "choice"), "ins": question["instructions"], "crit": question.get("criteria")}


ENCODINGS: Registry[Callable[..., "Item | None"]] = Registry("encoding")


@ENCODINGS.register("chunked-markers")
def encode_markers(tok, ex: dict, max_len: int, chunk_tokens: int, order: Sequence[int] | None = None) -> Item | None:
    """The Zeiger encoding. Returns None when the options do not fit `max_len` (never a truncated option list)."""
    q = to_internal(ex["question"])
    keys = option_keys(ex["question"])
    order = list(order) if order is not None else list(range(len(keys)))
    chunks, prefixes = build_chunks(tok, ex["state"], q, max_len, option_order=order, chunk_tokens=chunk_tokens)
    if sum(len(ms) for _, ms in chunks) != len(keys):
        return None
    label_key = str(ex["label"])
    if label_key not in keys:
        return None
    label = order.index(keys.index(label_key))
    return Item(chunks=chunks, prefix_lens=prefixes, keys=[keys[i] for i in order], qtype=QTYPES.get(q["t"], 0),
                label=label, ex_id=str(ex.get("id", "")), cat=ex.get("_cat", ""))


def collate(items: Sequence[Item], pad_id: int, bucket: int = PAD_MULTIPLE) -> dict[str, Any]:
    """Items -> one batch of chunk rows, right-padded to a multiple of `bucket`.

    input_ids/attention_mask/marker_pos/prefix_len are per CHUNK; marker_mask/qtype/label per QUESTION;
    q_index[i, j] = flat position of option j of question i among all chunk marker states ([C * max_markers]).
    """
    rows = [(qi, ids, markers, it.prefix_lens[ci]) for qi, it in enumerate(items) for ci, (ids, markers) in enumerate(it.chunks)]
    C = len(rows)
    L = -(-max(len(ids) for _, ids, _, _ in rows) // bucket) * bucket
    mmax = max(1, max(len(m) for _, _, m, _ in rows))
    n, kmax = len(items), max(it.n_markers for it in items)
    input_ids = torch.full((C, L), pad_id, dtype=torch.long)
    att = torch.zeros((C, L), dtype=torch.long)
    cpos = torch.zeros((C, mmax), dtype=torch.long)
    chunk_q = torch.zeros(C, dtype=torch.long)
    prefix_len = torch.zeros(C, dtype=torch.long)
    q_index = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    filled = [0] * n
    for c, (qi, ids, markers, prefix) in enumerate(rows):
        input_ids[c, : len(ids)] = torch.tensor(ids)
        att[c, : len(ids)] = 1
        chunk_q[c] = qi
        prefix_len[c] = prefix
        if markers:
            cpos[c, : len(markers)] = torch.tensor(markers)
            k0 = filled[qi]
            q_index[qi, k0: k0 + len(markers)] = c * mmax + torch.arange(len(markers))
            mmask[qi, k0: k0 + len(markers)] = True
            filled[qi] = k0 + len(markers)
    batch = {"input_ids": input_ids, "attention_mask": att, "marker_pos": cpos, "marker_mask": mmask, "q_index": q_index,
             "chunk_q": chunk_q, "prefix_len": prefix_len,
             "qtype": torch.tensor([it.qtype for it in items], dtype=torch.long),
             "label": torch.tensor([it.label for it in items], dtype=torch.long),
             "none_at": torch.tensor([it.keys.index("none") if "none" in it.keys else -1 for it in items], dtype=torch.long),
             "items": list(items)}
    return batch


def token_batches(items: Iterable[Item], budget: int) -> Iterable[list[Item]]:
    """Micro-batches under a padded-token budget, binned by longest chunk (<=512, <=1024, longer) so a short question
    does not share a batch with a full 2k page chunk. Padding is never attended, so this changes speed, never a result.
    An item larger than the budget gets a batch of its own."""
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
