"""The architecture: a causal decoder read as an encoder, with a decision head over option markers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ARCH = "qwen3-marker"
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
WINDOW = 32768
ATTENTION = "causal_flash"

_registered = False


def register_attention() -> None:
    """Mask-free causal attention: no L×L mask is built, so SDPA stays on its flash path."""
    global _registered
    if _registered:
        return
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    def causal_flash(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **_):
        repeats = query.shape[1] // key.shape[1]
        if repeats > 1:
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                             dropout_p=dropout if module.training else 0.0,
                                             is_causal=True, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register(ATTENTION, causal_flash)
    AttentionMaskInterface.register(ATTENTION, lambda *a, **k: None)
    _registered = True


class Zeiger(nn.Module):
    """Backbone plus head: one logit per option.

    The backbone sees each option after the state, the question and the options before it. The head is a small
    bidirectional transformer over the marker states alone, so an option is also judged against the options that
    follow it — at O(K²) in the number of options, never O(L²) in page length.
    """

    def __init__(self, encoder: nn.Module, head_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        width = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(width, max(1, width // 64), 2 * width, dropout,
                                           batch_first=True, norm_first=True)
        self.set_head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers else None
        self.type_emb = nn.Embedding(len(("choice", "score", "noul")), width)
        self.in_norm = nn.LayerNorm(width)
        self.scorer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, input_ids: Tensor, attention_mask: Tensor, marker_pos: Tensor, marker_mask: Tensor,
                qtype: Tensor, q_index: Tensor | None = None, **_) -> tuple[Tensor, None]:
        hidden = self.encoder(input_ids=input_ids, use_cache=False).last_hidden_state
        gather = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
        markers = torch.gather(hidden, 1, gather)
        if q_index is not None:
            markers = markers.reshape(-1, hidden.size(-1))[q_index]

        options = self.in_norm(markers.to(self.in_norm.weight.dtype)) + self.type_emb(qtype)[:, None, :]
        if self.set_head is not None:
            options = self.set_head(options, src_key_padding_mask=~marker_mask)
        logits = self.scorer(options).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4), None


def load(path: str | os.PathLike, max_len: int | None = None) -> tuple[Zeiger, object, dict]:
    """Load an exported checkpoint directory, or start from the base model when given a model id."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    register_attention()
    directory = Path(path)
    config_file = directory / "rl_agent_config.json"

    if config_file.exists():
        from safetensors.torch import load_file

        config = json.loads(config_file.read_text())
        if config.get("arch", ARCH) != ARCH:
            raise ValueError(f"{directory} holds a '{config['arch']}' model; this engine serves '{ARCH}'")
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(directory / "encoder"),
                                        attn_implementation=ATTENTION, dtype=torch.float32)
        model = Zeiger(encoder, config.get("head_layers", 2))
        model.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
    else:
        name = str(path) or BASE_MODEL
        tokenizer = AutoTokenizer.from_pretrained(name)
        encoder = AutoModel.from_pretrained(name, attn_implementation=ATTENTION, dtype=torch.float32)
        model = Zeiger(encoder)
        config = {"arch": ARCH, "encoder": name, "head_layers": 2, "chunk_tokens": 0,
                  "max_len": WINDOW, "temperature_by_options": {}}

    if max_len:
        config["max_len"] = max_len
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, config
