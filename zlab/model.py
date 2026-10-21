"""Backbone + head, initialised from base weights (Track B) or from a Zeiger checkpoint (Track A).

Checkpoints are read and written in Zeiger's export format (config.json or rl_agent_config.json, encoder/,
tokenizer/, model.safetensors with encoder.* + set_head/type_emb/in_norm/scorer keys), so the round-17 champion loads
here and a baseline-architecture result loads in the zeiger inference engine (`zeiger.model.load`).
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import torch
from torch import nn

from .backbones import BACKBONES, Backbone, attention_context, backbone_for_config
from .encoding import marker_token
from .heads import HEADS

ZEIGER_ARCH = "qwen3-marker"


class ZlabModel(nn.Module):
    def __init__(self, backbone: Backbone, head: nn.Module, attention_plan: dict | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = backbone.module  # parameter names encoder.* as in Zeiger checkpoints
        self.head = head
        self.attention_plan = attention_plan or {"default": "causal_nomask", "layers": {}}

    def forward(self, batch: dict) -> torch.Tensor:
        with attention_context(attention_mask=batch["attention_mask"], prefix_len=batch.get("prefix_len"),
                               batch=batch, plan=self.attention_plan):
            hidden = self.backbone.forward(batch["input_ids"], batch["attention_mask"])
        return self.head(hidden, batch)

    def freeze_lower(self, n_layers: int) -> None:
        """Freeze the embeddings and the lowest n layers (laya-browser's GPU profile: 8 of 28)."""
        if n_layers <= 0:
            return
        for p in self.backbone.embeddings().parameters():
            p.requires_grad = False
        for layer in self.backbone.layers()[:n_layers]:
            for p in layer.parameters():
                p.requires_grad = False

    def param_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        enc = [p for n, p in self.named_parameters() if n.startswith("encoder.") and p.requires_grad]
        head = [p for n, p in self.named_parameters() if n.startswith("head.") and p.requires_grad]
        return enc, head

    def state_dict_zeiger(self) -> dict[str, torch.Tensor]:
        out = {}
        for k, v in self.state_dict().items():
            if k.startswith("backbone."):
                continue  # same tensors as encoder.*
            out[k[len("head."):] if k.startswith("head.") else k] = v
        return out


def import_plugins(variant: dict) -> None:
    """variant.json "plugins": ["zlab.contrib.my_loss", ...] are imported before anything is built (registration)."""
    for mod in variant.get("plugins", []) or []:
        importlib.import_module(mod)


def checkpoint_config(directory: Path) -> dict | None:
    for name in ("config.json", "rl_agent_config.json"):
        f = directory / name
        if f.is_file():
            cfg = json.loads(f.read_text())
            if "arch" in cfg:
                return cfg
    return None


def build_head(variant: dict, width: int) -> nn.Module:
    spec = dict(variant.get("head") or {"kind": "marker-set"})
    return HEADS.get(spec.pop("kind", "marker-set"))(width, **spec)


def build_model(variant: dict, init: str, seed: int = 0) -> tuple[ZlabModel, object, dict]:
    """init: a checkpoint directory (Zeiger format) or "base" (backbone base weights, fresh head).
    Returns (model, tokenizer, info) with info = {init, base_model, backbone, missing_head_keys, ...}."""
    from transformers import AutoConfig

    import_plugins(variant)
    bspec = dict(variant.get("backbone") or {})
    plan = bspec.get("attention")  # None: the backbone's default (causal_nomask for decoders, bidirectional for encoders)
    info: dict = {"init": init}
    torch.manual_seed(seed)  # a fresh head starts from the same weights for the same seed
    ckpt = Path(init) if init and init != "base" else None
    if ckpt is not None:
        from safetensors.torch import load_file

        cfg = checkpoint_config(ckpt)
        if cfg is None:
            raise FileNotFoundError(f"{ckpt} is not a Zeiger checkpoint (no config.json / rl_agent_config.json with 'arch')")
        ecfg = AutoConfig.from_pretrained(ckpt / "encoder")
        kind = bspec.get("kind") or backbone_for_config(ecfg)
        plan = plan or dict(BACKBONES.get(kind).default_attention)
        backbone = BACKBONES.get(kind).from_config(ecfg, bspec)
        tok = BACKBONES.get(kind).load_tokenizer(str(ckpt / "tokenizer"))
        head = build_head(variant, backbone.hidden_size)
        model = ZlabModel(backbone, head, plan)
        state = load_file(ckpt / "model.safetensors")
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = backbone.module.load_state_dict(enc_state, strict=False)
        missing = [k for k in missing if "rotary" not in k]
        if missing or unexpected:
            raise RuntimeError(f"encoder weights do not match: missing {missing[:5]} unexpected {unexpected[:5]}")
        head_state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
        own = head.state_dict()
        loadable = {k: v for k, v in head_state.items() if k in own and own[k].shape == v.shape}
        head.load_state_dict(loadable, strict=False)
        info.update(base_model=cfg.get("encoder", ""), backbone=kind, checkpoint_arch=cfg.get("arch"),
                    head_loaded=sorted(loadable), head_fresh=sorted(k for k in own if k not in loadable),
                    head_unused=sorted(k for k in head_state if k not in loadable), checkpoint_config=cfg)
    else:
        kind = bspec.get("kind", "qwen3")
        plan = plan or dict(BACKBONES.get(kind).default_attention)
        base = bspec.get("base_model") or BACKBONES.get(kind).default_base
        backbone = BACKBONES.get(kind).from_pretrained(base, bspec)
        tok = BACKBONES.get(kind).load_tokenizer(base)
        head = build_head(variant, backbone.hidden_size)
        model = ZlabModel(backbone, head, plan)
        info.update(base_model=base, backbone=kind, head_fresh=sorted(head.state_dict()))
    mtok = marker_token(tok)
    marker = tok.convert_tokens_to_ids(mtok)
    if marker is None or marker == tok.unk_token_id:
        raise ValueError(f"tokenizer has no single {mtok} token; the backbone needs one added (and embeddings resized)")
    info["marker"] = mtok
    if ckpt is None:
        info["marker_init"] = init_marker_row(backbone, marker)
    return model, tok, info


def init_marker_row(backbone: Backbone, marker: int, floor: float = 0.1) -> dict:
    """Base weights only. A reserved token the base model never saw can have an embedding of ~0 (EuroBERT-610m: norm
    0.0034 vs median 0.225): every marker then enters as the same near-zero vector, and with frozen embeddings it never
    moves (EuroBERT run 922bd20: short 0.19, long 0.02). Such a row (norm < floor x median) is set to the mean
    embedding direction at the median norm. Rows with a usable norm (Qwen3 0.36x, Ettin 0.41x, mmBERT 0.83x median)
    are left unchanged."""
    with torch.no_grad():
        E = backbone.module.get_input_embeddings().weight.detach()  # same storage; the row is written in place
        norms = E.float().norm(dim=1)
        median = norms.median()
        before = float(norms[marker])
        if before >= floor * float(median):
            return {"changed": False, "norm": round(before, 4), "median": round(float(median), 4)}
        real = E.float()[norms >= floor * median].mean(0)
        E[marker] = (real / real.norm() * median).to(E.dtype)
    return {"changed": True, "norm_before": round(before, 4), "norm": round(float(E[marker].float().norm()), 4),
            "median": round(float(median), 4), "rule": "mean embedding direction x median norm"}


def save_checkpoint(model: ZlabModel, tok, directory: str | os.PathLike, variant: dict, extra: dict | None = None) -> None:
    """Zeiger export format (loadable by zeiger.model.load when the head is marker-set)."""
    from safetensors.torch import save_file

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict_zeiger().items()}, d / "model.safetensors")
    tok.save_pretrained(d / "tokenizer")
    model.encoder.config.save_pretrained(d / "encoder")
    head = variant.get("head") or {}
    baseline = head.get("kind", "marker-set") == "marker-set" and (variant.get("backbone") or {}).get("kind", "qwen3") == "qwen3"
    cfg = {"arch": ZEIGER_ARCH if baseline else f"zlab-{variant.get('name', 'variant')}",
           "encoder": (variant.get("backbone") or {}).get("base_model", ""), "head_layers": head.get("layers", 2),
           "chunk_tokens": (variant.get("encoding") or {}).get("chunk_tokens", 0), "max_len": 131072,
           "temperature_by_options": {}, "zlab_variant": variant, **(extra or {})}
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
