"""Training objective: cross-entropy over the option markers plus registered extra terms (registry `LOSSES`).

variant.json:

    "losses": {"ce": 1.0, "spherical": 0.5,
               "decision_reward": {"weight": 0.3, "cost_wrong_pick": 1.0, "cost_false_none": 0.35},
               "brier": 0.05}

A number is the term's weight; an object is {"weight": w, ...cfg}. Every term returns a SUM over the questions of
the micro-batch (the trainer divides by the questions of the optimizer step, like laya-browser), already signed as a
quantity to minimise; the step loss is ce_weight * CE + sum(weight * term). train.jsonl logs `ce` and every term's
per-question mean under `extra`.

Extension point:

    @LOSSES.register("kl_anchor")
    class KLAnchor(LossTerm):
        def setup(self, run): ...          # once, after the model is built (run.model, run.tokenizer, run.device,
                                           # run.variant, run.budget, run.track, run.init); e.g. load a frozen reference
        def __call__(self, ctx): ...        # ctx: LossContext -> scalar tensor (sum over questions)

Round 17 (the champion's recipe, = main's baseline): CE - 0.5 * spherical score + 0.3 * expected decision cost
+ 0.05 * Brier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .registry import Registry

LOSSES: Registry[type] = Registry("loss term")


@dataclass
class LossContext:
    logits: torch.Tensor  # [N, K] float32, -1e4 at padded options
    labels: torch.Tensor  # [N]
    batch: dict  # collated batch (on device): none_at, qtype, marker_mask, items, ...
    model: Any
    step: int
    cache: dict = field(default_factory=dict)

    @property
    def probs(self) -> torch.Tensor:
        if "probs" not in self.cache:
            self.cache["probs"] = torch.softmax(self.logits, -1)
        return self.cache["probs"]

    @property
    def p_label(self) -> torch.Tensor:
        return self.probs.gather(1, self.labels[:, None]).squeeze(1)


class LossTerm:
    def __init__(self, weight: float = 1.0, **cfg: Any) -> None:
        self.weight = float(weight)
        self.cfg = cfg

    def setup(self, run: Any) -> None:
        pass

    def __call__(self, ctx: LossContext) -> torch.Tensor:
        raise NotImplementedError


@LOSSES.register("spherical")
class Spherical(LossTerm):
    """Minus the spherical score p_label / ||p|| (the differentiable core of laya's RLCD reward)."""

    def __call__(self, ctx):
        q = ctx.probs
        return -(ctx.p_label / q.norm(dim=-1).clamp_min(1e-9)).sum()


@LOSSES.register("decision_reward")
class DecisionReward(LossTerm):
    """Minus the expected decision reward under the model's own distribution: acting on a wrong element costs
    cost_wrong_pick, a false 'none' costs cost_false_none, a right answer earns reward_correct(_none)."""

    def __call__(self, ctx):
        c = {"cost_wrong_pick": 1.0, "cost_false_none": 0.35, "reward_correct": 1.0, "reward_correct_none": 1.0, **self.cfg}
        q, labels = ctx.probs, ctx.labels
        none_at = ctx.batch["none_at"]
        has_none = none_at >= 0
        p_label = ctx.p_label
        p_none = q.gather(1, none_at.clamp(min=0)[:, None]).squeeze(1) * has_none
        label_is_none = has_none & (labels == none_at)
        picked_none = torch.where(label_is_none, torch.zeros_like(p_none), p_none)
        p_wrong = (1.0 - p_label - picked_none).clamp(min=0.0)
        reward = torch.where(label_is_none, c["reward_correct_none"] * p_label, c["reward_correct"] * p_label) \
            - c["cost_false_none"] * picked_none - c["cost_wrong_pick"] * p_wrong
        return -reward.sum()


@LOSSES.register("brier")
class Brier(LossTerm):
    def __call__(self, ctx):
        q = ctx.probs
        onehot = torch.zeros_like(q).scatter_(1, ctx.labels[:, None], 1.0)
        return ((q - onehot) ** 2 * ctx.batch["marker_mask"]).sum()


def build_losses(spec: dict | None) -> tuple[float, list[tuple[str, LossTerm]]]:
    """-> (ce weight, [(name, term)])."""
    spec = dict(spec or {"ce": 1.0})
    ce_w = float(spec.pop("ce", 1.0))
    terms = []
    for name, cfg in spec.items():
        if name.startswith("_"):
            continue
        cfg = {"weight": cfg} if isinstance(cfg, (int, float)) else dict(cfg)
        if not cfg.get("weight", 1.0):
            continue
        terms.append((name, LOSSES.get(name)(**cfg)))
    return ce_w, terms


def compute(logits: torch.Tensor, labels: torch.Tensor, batch: dict, model: Any, step: int, ce_weight: float,
            terms: list[tuple[str, LossTerm]]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """-> (total loss sum, ce sum, {term: raw value sum}) for one micro-batch."""
    ce = F.cross_entropy(logits, labels, reduction="sum")
    ctx = LossContext(logits=logits, labels=labels, batch=batch, model=model, step=step)
    total = ce_weight * ce
    parts = {}
    for name, term in terms:
        v = term(ctx)
        parts[name] = v.detach()
        total = total + term.weight * v
    return total, ce.detach(), parts
