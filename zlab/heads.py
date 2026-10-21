"""Decision heads: backbone hidden states -> one logit per option (registry `HEADS`).

A head is an nn.Module built as `cls(width, **cfg)` (cfg = variant.json "head" minus "kind") with

    forward(hidden [C, L, D], batch) -> logits [N, K] (float32, -1e4 at padded options)

`hidden` holds every chunk row's states (not only the markers), so a head may read the page tokens too; `batch` is
the collated batch (marker_pos, q_index, marker_mask, qtype, chunk_q, prefix_len, attention_mask, items).

`marker-set` is Zeiger's head, parameter for parameter (names set_head / type_emb / in_norm / scorer match the
round-17 checkpoint): the marker states, LayerNorm + a question-type embedding, a 2-layer bidirectional
transformer over the K option states of a question, then an MLP scorer.
"""

from __future__ import annotations

import torch
from torch import nn

from .registry import Registry

HEADS: Registry[type] = Registry("head")
N_QTYPES = 3


def gather_markers(hidden: torch.Tensor, batch: dict) -> torch.Tensor:
    """[C, L, D] -> [N, K, D]: the hidden state at every option marker, per question (q_index maps chunks to questions)."""
    marker_pos = batch["marker_pos"]
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
    m = torch.gather(hidden, 1, idx)
    return m.reshape(-1, m.size(-1))[batch["q_index"]]


@HEADS.register("marker-set")
class MarkerSetHead(nn.Module):
    def __init__(self, width: int, layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        layer = nn.TransformerEncoderLayer(width, max(1, width // 64), 2 * width, dropout, batch_first=True, norm_first=True)
        self.set_head = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False) if layers > 0 else None
        self.type_emb = nn.Embedding(N_QTYPES, width)
        self.in_norm = nn.LayerNorm(width)
        self.scorer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def option_states(self, hidden: torch.Tensor, batch: dict) -> torch.Tensor:
        """[N, K, D] option states after the set module (before the scorer): a branch can reuse this."""
        m = gather_markers(hidden, batch).to(self.in_norm.weight.dtype)
        m = self.in_norm(m) + self.type_emb(batch["qtype"])[:, None, :]
        if self.set_head is not None:
            m = self.set_head(m, src_key_padding_mask=~batch["marker_mask"])
        return m

    def forward(self, hidden: torch.Tensor, batch: dict) -> torch.Tensor:
        logits = self.scorer(self.option_states(hidden, batch)).squeeze(-1).float()
        return logits.masked_fill(~batch["marker_mask"], -1e4)
