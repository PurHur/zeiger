"""memory-head (ZEIGER_NEXT decision `memory-head`, after Cloudflare Clef's evidence-routing stage).

Zeiger's set head only sees one marker vector per option, so options in different 2k chunks meet only through those
vectors. Clef lets every option cross-attend to the whole prompt's hidden states before its field layers
(`clones/cloudflare-clef.md`, "Readout head", stage 1). This head keeps the champion's head and adds, before the set
layers:

1. per chunk, `memory_tokens` (4) learned queries attention-pool that chunk's real token states into 4 summary tokens
   (single-head dot-product pooling, scores = h . q / sqrt(D), padding masked), then Linear + LayerNorm;
2. one cross-attention layer (`memory_heads` = 8 heads): every option vector of a question attends to the summary
   tokens of ALL chunks of that question (never another question's: Jev's question isolation is kept);
   m <- m + out_proj(MHA(LN(m), mem, mem)), with out_proj zero-initialised, so at step 0 the logits are exactly the
   champion's (Track A continues from the champion, not from a random head).

The backbone already hands every chunk row's states to the head (`hidden` [C, L, D]), so no model change is needed.
Cost: pooling is O(C x L x 4), the cross-attention O(N x K x 4C); nothing grows with L^2.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from zlab.heads import HEADS, MarkerSetHead, gather_markers


def question_memory(summaries: torch.Tensor, chunk_q: torch.Tensor, n_questions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """summaries [C, S, D] (per chunk) -> memory [N, Cmax*S, D] and its key padding mask [N, Cmax*S] (True = padding).
    Chunk rows of one question are consecutive in a collated batch (zlab.encoding.collate)."""
    C, S, D = summaries.shape
    counts = torch.bincount(chunk_q, minlength=n_questions)
    first = torch.cumsum(counts, 0) - counts
    local = torch.arange(C, device=chunk_q.device) - first[chunk_q]
    cmax = int(counts.max()) if C else 1
    mem = summaries.new_zeros(n_questions, cmax * S, D)
    pad = torch.ones(n_questions, cmax * S, dtype=torch.bool, device=summaries.device)
    slot = (local[:, None] * S + torch.arange(S, device=chunk_q.device)[None, :])  # [C, S]
    qi = chunk_q[:, None].expand(-1, S)
    mem[qi, slot] = summaries
    pad[qi, slot] = False
    return mem, pad


@HEADS.register("memory-head")
class MemoryHead(MarkerSetHead):
    def __init__(self, width: int, layers: int = 2, dropout: float = 0.1, memory_tokens: int = 4, memory_heads: int = 8,
                 memory_dropout: float = 0.0, **cfg):
        super().__init__(width, layers=layers, dropout=dropout, **cfg)
        self.memory_tokens = memory_tokens
        self.mem_queries = nn.Parameter(torch.randn(memory_tokens, width) * 0.02)  # ~uniform pooling at init
        self.mem_proj = nn.Linear(width, width)
        self.mem_norm = nn.LayerNorm(width)
        self.mem_q_norm = nn.LayerNorm(width)
        self.mem_attn = nn.MultiheadAttention(width, memory_heads, dropout=memory_dropout, batch_first=True)
        nn.init.zeros_(self.mem_attn.out_proj.weight)
        nn.init.zeros_(self.mem_attn.out_proj.bias)

    def chunk_summaries(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """[C, L, D] -> [C, S, D]: attention pooling of each chunk's real tokens with the learned queries."""
        q = self.mem_queries.to(hidden.dtype)
        scores = torch.einsum("cld,sd->csl", hidden, q) / math.sqrt(hidden.size(-1))
        scores = scores.float().masked_fill(~attention_mask.bool()[:, None, :], float("-inf"))
        weights = torch.softmax(scores, -1).to(hidden.dtype)
        pooled = torch.einsum("csl,cld->csd", weights, hidden)
        return self.mem_norm(self.mem_proj(pooled.to(self.mem_proj.weight.dtype)))

    def option_states(self, hidden: torch.Tensor, batch: dict) -> torch.Tensor:
        m = gather_markers(hidden, batch).to(self.in_norm.weight.dtype)
        m = self.in_norm(m) + self.type_emb(batch["qtype"])[:, None, :]
        summaries = self.chunk_summaries(hidden, batch["attention_mask"])
        mem, mem_pad = question_memory(summaries, batch["chunk_q"], m.size(0))
        routed, _ = self.mem_attn(self.mem_q_norm(m), mem.to(m.dtype), mem.to(m.dtype), key_padding_mask=mem_pad,
                                  need_weights=False)
        m = m + routed
        if self.set_head is not None:
            m = self.set_head(m, src_key_padding_mask=~batch["marker_mask"])
        return m


def added_params(width: int, memory_tokens: int = 4) -> int:
    """Parameters this head adds over marker-set (for the describe block / report)."""
    h = MemoryHead(width, layers=0, memory_tokens=memory_tokens)
    base = MarkerSetHead(width, layers=0)
    return sum(p.numel() for p in h.parameters()) - sum(p.numel() for p in base.parameters())
