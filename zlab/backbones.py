"""Backbones: the transformer that reads the tokens (registry), and the attention hooks they run with.

Registry `BACKBONES` (name -> Backbone subclass). A backbone knows how to build its module from base weights or from
a saved config, which modules are its embeddings and its layers (freezing), its hidden size, its default attention
plan and which reserved token of its tokenizer serves as the option marker. Implemented:

| kind | family | default attention | marker |
|---|---|---|---|
| `qwen3` | dense causal Qwen3 (0.6B/1.7B-Base, Qwen3-Reranker-0.6B body) | `causal_nomask` | `<\\|object_ref_end\\|>` |
| `qwen3.5` | Gated DeltaNet + full-attention hybrid, text model only | `causal_nomask` (full-attn layers) | `<\\|object_ref_end\\|>` |
| `eurobert` | EuroBERT (Llama-style bidirectional encoder, native in transformers 5) | `bidirectional` | `<\\|reserved_special_token_2\\|>` |
| `mmbert` / `ettin` | ModernBERT encoders (global + local +-64 window layers) | `bidirectional` | `<unused0>` / `[unused0]` |
| `gemma` | Gemma 4 text model (sliding 512 + full layers, per-layer embeddings, KV sharing) | `causal_nomask` (window-aware) | `<unused0>` |

Attention hook. Every attention layer of an HF backbone loaded here calls `zlab_attention`, which dispatches per
layer to a function from the `ATTENTION` registry, chosen by variant.json:

    "backbone": {"attention": {"default": "causal_nomask", "layers": {"24": "my_hook", "25": "my_hook"}}}

An attention function has the signature

    fn(module, query, key, value, *, dropout, scaling, ctx) -> attn_output [B, L, H, D]

query/key/value are [B, H, L, D] (KV heads not yet repeated); `ctx` is the `AttentionContext` of the current
forward pass: `ctx.attention_mask` ([B, L], 1 = real token), `ctx.prefix_len` ([B], state+question tokens of every
chunk), `ctx.batch` (the whole collated batch) and `ctx.layer_idx`. The default `causal_nomask` is Zeiger's mask-free
causal SDPA: rows are right-padded and attention is causal, so a real token never attends to padding and SDPA stays
on its flash path (a materialised L x L mask leaves it: measured 34.6 s for one 8k training sequence).
A non-causal hook must mask padding itself (ctx.attention_mask).

Local (sliding-window) layers: the adapter stores the window on the attention module as `module.zlab_window =
(left, right)` (keys j with i-left <= j <= i+right). `causal_nomask` and `bidirectional` honour it with a blocked
band attention (keys of the neighbouring blocks only, a small [block x 3 block] mask), so a 2k chunk never builds an
L x L mask. A branch hook that ignores `zlab_window` turns local layers into global ones.

Non-causal attention without a mask: `bidirectional` runs SDPA per group of rows with the same real length on the
sliced real tokens (no mask, flash path); outputs at padding positions are 0 (never read: padding keys are never
attended). For Qwen3.5 the hook runs on the full-attention layers only (DeltaNet layers are causal recurrences and
never call it), so `{"default": "noncausal_full_attention"}` is pplx-decider v1.1 / vLLM-SR d3's "noncausal full
attention, DeltaNet stays causal" (padding mask on those layers only).

Gradient checkpointing recomputes layers during backward, outside the forward's `attention_context` (and on GPU in
autograd's own thread): while autograd runs a backward, `current_context()` falls back to the latest forward's
context (`_last`), so the recomputation uses the same plan and padding mask as the forward.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from .registry import Registry

ATTN_NAME = "zlab_attn"  # transformers >= 5 refuses locally registered names containing "flash"

ATTENTION: Registry[Callable[..., torch.Tensor]] = Registry("attention function")
BACKBONES: Registry[type] = Registry("backbone")


@dataclass
class AttentionContext:
    attention_mask: torch.Tensor | None = None
    prefix_len: torch.Tensor | None = None
    batch: dict | None = None
    plan: dict | None = None  # {"default": name, "layers": {idx: name}}
    layer_idx: int | None = None
    cache: dict | None = None  # per-forward scratch (row groups), never inherited by a nested context


_local = threading.local()
# The context of the latest forward pass, process-wide. Gradient checkpointing recomputes each layer during
# backward(), i.e. after the forward's `with attention_context(...)` has exited and (on GPU) on autograd's device
# thread, where the thread-local context is empty: without this fallback the recomputation silently ran the default
# causal_nomask instead of the planned hook, so a hook's backward did not match its forward. The fallback is used
# only while autograd is executing a backward pass (training backpropagates each micro-batch right after its forward).
_last: list = [None]


def _in_backward() -> bool:
    try:
        return torch._C._current_graph_task_id() != -1
    except AttributeError:  # pragma: no cover - older torch
        return False


def current_context() -> AttentionContext:
    ctx = getattr(_local, "ctx", None)
    if ctx is not None and ctx.plan is not None:
        return ctx
    if _last[0] is not None and _in_backward():
        return _last[0]
    if ctx is None:
        ctx = _local.ctx = AttentionContext()
    return ctx


@contextlib.contextmanager
def attention_context(**kw: Any):
    prev = getattr(_local, "ctx", None)
    inherited = {k: v for k, v in (prev.__dict__ if prev else {}).items() if k != "cache"}
    _local.ctx = AttentionContext(**{**inherited, **kw, "cache": {}})
    _last[0] = _local.ctx
    try:
        yield _local.ctx
    finally:
        _local.ctx = prev


def repeat_kv(key: torch.Tensor, value: torch.Tensor, n_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    rep = n_heads // key.shape[1]
    if rep > 1:
        key, value = key.repeat_interleave(rep, dim=1), value.repeat_interleave(rep, dim=1)
    return key, value


def banded_sdpa(query, key, value, left: int, right: int, valid: torch.Tensor | None = None, *, dropout=0.0,
                scaling=None) -> torch.Tensor:
    """Attention of query i to keys j with i-left <= j <= i+right (and valid[b, j]) without an L x L mask.
    q/k/v [B, H, L, D] (KV repeated) -> [B, H, L, D]. The sequence is cut into blocks of max(left, right) tokens;
    each query block attends to its own and the neighbouring key blocks through a [block, W] mask. A query always
    sees itself, so padded rows never become all-masked (NaN)."""
    B, H, L, D = query.shape
    blk = max(1, min(L, max(left, right)))
    nl, nr = -(-left // blk), -(-right // blk)
    nb = -(-L // blk)
    Lp, W = nb * blk, (nl + 1 + nr) * blk
    dev = query.device
    qb = F.pad(query, (0, 0, 0, Lp - L)).view(B, H, nb, blk, D).transpose(1, 2).reshape(B * nb, H, blk, D)

    def windows(x):  # [B, H, L, D] -> [B*nb, H, W, D]
        xp = F.pad(x, (0, 0, nl * blk, Lp - L + nr * blk))
        return xp.unfold(2, W, blk).permute(0, 2, 1, 4, 3).reshape(B * nb, H, W, D)

    kb, vb = windows(key), windows(value)
    qi = (torch.arange(nb, device=dev)[:, None] * blk + torch.arange(blk, device=dev))[:, :, None]  # [nb, blk, 1]
    kj = (torch.arange(nb, device=dev)[:, None] * blk - nl * blk + torch.arange(W, device=dev))[:, None, :]  # [nb, 1, W]
    rel = kj - qi
    allowed = (rel >= -left) & (rel <= right) & (kj >= 0) & (kj < L)  # [nb, blk, W]
    if valid is not None:
        vw = F.pad(valid.to(torch.bool), (nl * blk, Lp - L + nr * blk)).unfold(1, W, blk)  # [B, nb, W]
        allowed = allowed[None] & vw[:, :, None, :]
    else:
        allowed = allowed[None].expand(B, -1, -1, -1)
    allowed = allowed | (rel == 0)[None]
    out = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=allowed.reshape(B * nb, 1, blk, W), dropout_p=dropout,
                                         scale=scaling)
    return out.view(B, nb, H, blk, D).transpose(1, 2).reshape(B, H, Lp, D)[:, :, :L]


def _row_groups(mask: torch.Tensor, cache_on: AttentionContext | None) -> list[tuple[torch.Tensor, int]] | None:
    """Right-padded rows grouped by real length: [(row indices, length)], or None when the mask is not a right-padding
    prefix mask (then the caller falls back to a key-padding mask). Cached per forward on the context."""
    cache = cache_on.cache if cache_on is not None else None
    cached = cache.get("row_groups") if cache is not None else None
    if cached is not None and cached[0] is mask:
        return cached[1]
    m = mask.to(torch.bool)
    lens = m.sum(1)
    pos = torch.arange(m.shape[1], device=m.device)[None]
    groups = None
    if bool(((pos < lens[:, None]) == m).all()):
        lens_l = lens.tolist()
        groups = [(torch.tensor([i for i, n in enumerate(lens_l) if n == L], device=m.device), int(L))
                  for L in sorted(set(lens_l)) if L > 0]
    if cache is not None:
        cache["row_groups"] = (mask, groups)
    return groups


def bidirectional_rows(query, key, value, valid: torch.Tensor | None, *, dropout=0.0, scaling=None,
                       ctx: AttentionContext | None = None) -> torch.Tensor:
    """Full attention over the real tokens of each row, SDPA without a mask per group of equal-length rows (flash
    path). q/k/v [B, H, L, D] (KV repeated) -> [B, H, L, D], 0 at padding positions."""
    L = query.shape[2]
    if valid is None or bool(valid.all()):
        return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout, is_causal=False, scale=scaling)
    groups = _row_groups(valid, ctx)
    if groups is None:  # not right-padded: key-padding mask (leaves the flash path, still exact)
        mask = valid.to(torch.bool)[:, None, None, :]
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling)
    outs, order = [], []
    for rows, n in groups:
        o = F.scaled_dot_product_attention(query[rows, :, :n], key[rows, :, :n], value[rows, :, :n], dropout_p=dropout,
                                           is_causal=False, scale=scaling)
        outs.append(F.pad(o, (0, 0, 0, L - n)) if n < L else o)
        order.append(rows)
    order = torch.cat(order)
    out = torch.cat(outs) if len(outs) > 1 else outs[0]
    if len(order) == query.shape[0]:
        return out[torch.argsort(order)]
    full = query.new_zeros(query.shape)  # rows without any real token
    return full.index_copy(0, order, out)


@ATTENTION.register("causal_nomask")
def causal_nomask(module, query, key, value, *, dropout=0.0, scaling=None, ctx=None):
    key, value = repeat_kv(key, value, query.shape[1])
    window = getattr(module, "zlab_window", None)
    if window is not None and window[0] + 1 < query.shape[2]:
        # causal local layer (Gemma sliding): j in [i-left, i]; right padding is never attended by real tokens
        out = banded_sdpa(query, key, value, window[0], 0, None, dropout=dropout, scaling=scaling)
    else:
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=True, scale=scaling)
    return out.transpose(1, 2).contiguous()


@ATTENTION.register("bidirectional_padmask")
def bidirectional_padmask(module, query, key, value, *, dropout=0.0, scaling=None, ctx=None):
    """Full (non-causal) attention over the real tokens of each row. Reference hook for branches; builds a [B,1,1,L]
    key mask, which costs the flash path on ROCm — measure before using it on long chunks. Ignores local windows."""
    key, value = repeat_kv(key, value, query.shape[1])
    mask = None
    if ctx is not None and ctx.attention_mask is not None:
        mask = ctx.attention_mask.to(torch.bool)[:, None, None, :]
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout, is_causal=False, scale=scaling)
    return out.transpose(1, 2).contiguous()


@ATTENTION.register("bidirectional")
def bidirectional(module, query, key, value, *, dropout=0.0, scaling=None, ctx=None):
    """Non-causal attention over the real tokens of each row (encoders' default). Global layers: unmasked SDPA per
    equal-length row group (flash path); local layers (`module.zlab_window`): blocked band attention + key padding.
    Exactly equal to a key-padding mask (+ window) at every real token (tests/test_backbones.py)."""
    key, value = repeat_kv(key, value, query.shape[1])
    valid = ctx.attention_mask if ctx is not None else None
    window = getattr(module, "zlab_window", None)
    if window is not None and max(window) + 1 < query.shape[2]:
        out = banded_sdpa(query, key, value, window[0], window[1], valid, dropout=dropout, scaling=scaling)
    else:
        out = bidirectional_rows(query, key, value, valid, dropout=dropout, scaling=scaling, ctx=ctx)
    return out.transpose(1, 2).contiguous()


# pplx-decider v1.1 / vLLM-SR d3 on Qwen3.5: the hook only runs on full-attention layers, so this is "noncausal on
# full-attention layers only, DeltaNet stays causal". Same function, its own name for meta.json / the dashboard.
ATTENTION.register("noncausal_full_attention")(bidirectional)


def zlab_attention(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **_):
    """The one function transformers calls; dispatches to the registered function planned for this layer."""
    ctx = current_context()  # in a checkpoint recomputation (backward): the latest forward's context
    plan = ctx.plan or {"default": "causal_nomask", "layers": {}}
    idx = getattr(module, "layer_idx", None)
    name = (plan.get("layers") or {}).get(str(idx), plan.get("default", "causal_nomask"))
    ctx.layer_idx = idx
    fn = ATTENTION.get(name)
    out = fn(module, query, key, value, dropout=dropout if module.training else 0.0, scaling=scaling, ctx=ctx)
    return out, None


_registered = False


def register_hf_attention() -> None:
    global _registered
    if _registered:
        return
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register(ATTN_NAME, zlab_attention)
    AttentionMaskInterface.register(ATTN_NAME, lambda *a, **k: None)  # never build an L x L mask
    _registered = True


# ------------------------------------------------------------------------------------------------- backbones

MARKER_CANDIDATES = (
    "<|object_ref_end|>",  # Qwen3 / Qwen3.5: reserved vision-grounding token, never in plain text
    "<|reserved_special_token_2|>",  # EuroBERT (Llama-3 vocabulary; reserved 0/1 became <|mask|>/<|parallel_sep|>)
    "<unused0>",  # Gemma tokenizers (mmBERT, Gemma 4)
    "[unused0]",  # ModernBERT / Ettin
)


def pick_marker(tok) -> str:
    """The first reserved token of MARKER_CANDIDATES that the tokenizer maps to one id of its own (not unk)."""
    vocab = tok.get_vocab()
    for cand in MARKER_CANDIDATES:
        i = vocab.get(cand)
        if i is not None and i != tok.unk_token_id:
            return cand
    raise ValueError("tokenizer has none of the reserved marker tokens %s; add one and resize the embeddings"
                     % (MARKER_CANDIDATES,))


class Backbone:
    """Wraps construction and introspection of one family. `module` is an nn.Module with
    forward(input_ids=..., use_cache=False) -> object with .last_hidden_state [B, L, D]."""

    kind = "base"
    default_base = ""
    default_attention = {"default": "causal_nomask", "layers": {}}
    causal = True

    def __init__(self, module: nn.Module, spec: dict) -> None:
        self.module = module
        self.spec = spec
        self.annotate_windows()

    # construction --------------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, name: str, spec: dict) -> "Backbone":
        raise NotImplementedError

    @classmethod
    def from_config(cls, config, spec: dict) -> "Backbone":
        raise NotImplementedError

    @classmethod
    def load_tokenizer(cls, name: str):
        """The backbone's own tokenizer; `tok.zlab_marker` = the reserved token used as option marker (encoding.py
        replaces it in text and appends its id after every option)."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        if tok.eos_token_id is None and getattr(tok, "sep_token", None) is not None:
            tok.eos_token = tok.sep_token  # ModernBERT/Ettin: [SEP] ends a sequence
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.zlab_marker = pick_marker(tok)
        return tok

    def annotate_windows(self) -> None:
        """Set `zlab_window = (left, right)` on local-attention modules (none for global-only families)."""

    # introspection -------------------------------------------------------------------------------------
    @property
    def hidden_size(self) -> int:
        return self.module.config.hidden_size

    def embeddings(self) -> nn.Module:
        return self.module.embed_tokens

    def layers(self) -> list[nn.Module]:
        return list(self.module.layers)

    def config(self):
        return self.module.config

    def enable_checkpointing(self) -> None:
        self.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # attention_mask deliberately not given to the model (see module docstring); hooks read it from the context
        return self.module(input_ids=input_ids, use_cache=False).last_hidden_state


def _hf_pretrained(model_cls, name: str, **kw):
    register_hf_attention()
    return model_cls.from_pretrained(name, attn_implementation=ATTN_NAME, dtype=torch.float32, **kw)


def _hf_from_config(model_cls, config):
    register_hf_attention()
    config._attn_implementation = ATTN_NAME
    return model_cls._from_config(config, attn_implementation=ATTN_NAME, dtype=torch.float32)


@BACKBONES.register("qwen3")
class Qwen3(Backbone):
    """Dense causal Qwen3, AutoModel = the decoder stack without the LM head. Zeiger: Qwen/Qwen3-0.6B-Base. Same class
    for Qwen/Qwen3-1.7B-Base (28 layers, d 2048) and Qwen/Qwen3-Reranker-0.6B (a fine-tuned Qwen3ForCausalLM with
    exactly Qwen3-0.6B's shape; its body loads here, the yes/no LM head is dropped; vocab 151,669 rows = its
    tokenizer, eos <|im_end|>)."""

    kind = "qwen3"
    default_base = "Qwen/Qwen3-0.6B-Base"

    @classmethod
    def from_pretrained(cls, name, spec):
        from transformers import AutoModel

        return cls(_hf_pretrained(AutoModel, name), spec)

    @classmethod
    def from_config(cls, config, spec):
        from transformers import AutoModel

        register_hf_attention()
        return cls(AutoModel.from_config(config, attn_implementation=ATTN_NAME, dtype=torch.float32), spec)


@BACKBONES.register("qwen3.5")
class Qwen35(Backbone):
    """Qwen3.5 hybrid (3 of 4 layers Gated DeltaNet linear attention, every 4th full attention). The base checkpoints
    are multimodal; only the text model is loaded. The attention hook applies to the full-attention layers only; the
    DeltaNet layers are causal recurrences and unaffected by right padding. Plan
    `{"default": "noncausal_full_attention"}` = pplx-decider v1.1 / vLLM-SR d3 (full-attention layers bidirectional
    over the real tokens, DeltaNet causal)."""

    kind = "qwen3.5"
    default_base = "Qwen/Qwen3.5-0.8B-Base"

    @classmethod
    def from_pretrained(cls, name, spec):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        lm = _hf_pretrained(Qwen3_5ForCausalLM, name)
        module = lm.model
        del lm
        return cls(module, spec)

    @classmethod
    def from_config(cls, config, spec):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        text = getattr(config, "text_config", None) or config
        return cls(_hf_from_config(Qwen3_5TextModel, text), spec)

    def full_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.module.config.layer_types) if t == "full_attention"]


@BACKBONES.register("eurobert")
class EuroBert(Backbone):
    """EuroBERT (210m/610m/2.1b): Llama-style bidirectional encoder, RoPE theta 250k, 8k context. Native in
    transformers 5 (`EuroBertModel`), so the Hub repo's `modeling_eurobert.py` (auto_map, trust_remote_code) is not
    executed; the checkpoint's `model.*` keys load into the native class. All layers global."""

    kind = "eurobert"
    default_base = "EuroBERT/EuroBERT-610m"
    default_attention = {"default": "bidirectional", "layers": {}}
    causal = False

    @classmethod
    def from_pretrained(cls, name, spec):
        from transformers import EuroBertModel

        return cls(_hf_pretrained(EuroBertModel, name), spec)

    @classmethod
    def from_config(cls, config, spec):
        from transformers import EuroBertModel

        return cls(_hf_from_config(EuroBertModel, config), spec)


class ModernBert(Backbone):
    """ModernBERT architecture (mmBERT, Ettin encoders): every 3rd layer global, the others local with a +-64 token
    window (config.sliding_window = local_attention // 2; HF semantics |i - j| <= 64), GeGLU, RoPE (160k), final norm.
    `embeddings()` = token embedding + its LayerNorm."""

    default_attention = {"default": "bidirectional", "layers": {}}
    causal = False

    @classmethod
    def from_pretrained(cls, name, spec):
        from transformers import ModernBertModel

        return cls(_hf_pretrained(ModernBertModel, name), spec)

    @classmethod
    def from_config(cls, config, spec):
        from transformers import ModernBertModel

        return cls(_hf_from_config(ModernBertModel, config), spec)

    def annotate_windows(self) -> None:
        half = int(self.module.config.sliding_window)
        for layer in self.module.layers:
            if getattr(layer.attn, "sliding_window", None):
                layer.attn.zlab_window = (half, half)

    def embeddings(self) -> nn.Module:
        return self.module.embeddings


@BACKBONES.register("mmbert")
class MmBert(ModernBert):
    """mmBERT (jhu-clsp, MIT): ModernBERT on 1,800+ languages, Gemma 2 tokenizer (256k vocab)."""

    kind = "mmbert"
    default_base = "jhu-clsp/mmBERT-base"


@BACKBONES.register("ettin")
class Ettin(ModernBert):
    """Ettin encoders (jhu-clsp, MIT): ModernBERT recipe, paired with same-data Ettin decoders, OLMo tokenizer (50k)."""

    kind = "ettin"
    default_base = "jhu-clsp/ettin-encoder-400m"


@BACKBONES.register("gemma")
class Gemma(Backbone):
    """Gemma 4 text model (google/gemma-4-E2B, Apache-2.0): 35 layers, 4 of 5 sliding-window (512, causal) and every
    5th full attention (head_dim 512), per-layer token embeddings (PLE, 262k x 35 x 256 = 2.35B params) and the last
    20 layers sharing K/V of earlier layers. The Hub checkpoint is the multimodal Gemma4ForConditionalGeneration; only
    `model.language_model.*` is loaded (key_mapping), vision/audio towers are skipped.

    spec "ple_dtype" (default "bfloat16"): dtype of the per-layer embedding table, which `freeze_lower` freezes with the
    token embedding; bf16 saves 4.7 GB of the 32 GB cap. `embeddings()` = token embedding + PLE table."""

    kind = "gemma"
    default_base = "google/gemma-4-E2B"

    @classmethod
    def _text_config(cls, config):
        return getattr(config, "text_config", None) or config

    @classmethod
    def from_pretrained(cls, name, spec):
        from transformers import AutoConfig, Gemma4TextModel

        text = cls._text_config(AutoConfig.from_pretrained(name))
        module = _hf_pretrained(Gemma4TextModel, name, config=text,
                                key_mapping={r"^model\.language_model\.": ""})
        return cls(module, spec)

    @classmethod
    def from_config(cls, config, spec):
        from transformers import Gemma4TextModel

        return cls(_hf_from_config(Gemma4TextModel, cls._text_config(config)), spec)

    def __init__(self, module, spec):
        super().__init__(module, spec)
        ple = getattr(module, "embed_tokens_per_layer", None)
        dtype = (spec or {}).get("ple_dtype", "bfloat16")
        if ple is not None and dtype:
            ple.to(getattr(torch, dtype))

    def annotate_windows(self) -> None:
        for layer in self.module.layers:
            sw = getattr(layer.self_attn, "sliding_window", None)
            if sw:  # HF: kv_idx > q_idx - sliding_window -> i-(sw-1) <= j <= i
                layer.self_attn.zlab_window = (int(sw) - 1, int(sw) - 1)

    def embeddings(self) -> nn.Module:
        mods = [self.module.embed_tokens]
        if getattr(self.module, "embed_tokens_per_layer", None) is not None:
            mods.append(self.module.embed_tokens_per_layer)
        return nn.ModuleList(mods)


def backbone_for_config(config) -> str:
    """Backbone kind for a saved HF encoder config (model_type)."""
    mt = getattr(config, "model_type", "")
    if mt.startswith("qwen3_5"):
        return "qwen3.5"
    if mt == "qwen3":
        return "qwen3"
    if mt.startswith("gemma4"):
        return "gemma"
    if mt == "modernbert":
        return "ettin" if getattr(config, "vocab_size", 0) < 100_000 else "mmbert"
    return mt
