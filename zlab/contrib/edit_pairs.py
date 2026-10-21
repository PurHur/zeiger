"""Minimal edit pairs that flip the answer (decision `edit-pairs`, after Bespoke Nimble's contrastive curation).

Nimble's idea is in the data: pairs of examples where a small edit to one fact flips the label, labels computed by
code from the facts, so the model cannot answer from the surface. Zeiger's precondition shortcut is that failure:
it picks the element the instruction names even when that element is disabled. A pair here is the same page and
instruction twice, differing in ONE precondition attribute of ONE element of the element table, with the label
computed by a solver: in the twin the label moves from the target to the blocking element, or to "none".

What precondition data exists (inspected bench/data/train.jsonl and laya-browser scripts/build_logic.py):
- `structured-logic` rows with meta.task == "logic-precond" (~1.4k of 60k train rows) carry a real precondition
  table: `button "Pay now" — disabled until VAT number is filled in`, `input "VAT number" — empty; usable once
  Street address is checked`, `checkbox "Email" — checked`, `dropdown "Username" — nothing selected`, sold-out
  fields. build_logic.py already pairs some of them (the deepest field done -> label moves up). Here any single
  state flip that changes the solver's answer is a candidate, preferring met -> unmet requirements (the label moves
  DOWN to the blocking element), which build_logic never generates.
- Crawled browser rows (mind2web, mt-mind2web, jevpilot-trace) have NO element states in their tables: laya's
  describe() emits role, name, placeholder, title, href, #id, region and "(below the fold)" only ("Required" appears
  only inside accessible names). So browser twins put a synthetic precondition onto the real page in the
  logic-precond wording: the pair's original says `button "Search" — disabled until City or Airport is filled in`
  plus `textbox "City or Airport" — filled in` (label: the target), the twin flips that field to `— empty` (label:
  the field). Targets without a usable field on the page get `— enabled` / `— disabled` (twin label: none).
  Because the answer to "which element does the instruction refer to?" does not change when an element is
  disabled, both halves of a browser pair ask the action question instead: ACTION_Q below.

Budget: a pair REPLACES sampling, it does not extend the run. When an example is chosen, the transform returns the
original (browser: re-asked) now and the twin in place of the NEXT example of the stream, which is dropped. Steps
still fill to tokens_per_step, so total training tokens are those of the budget. The twin copies the original's
option order (list this transform after shuffle_options) so the two differ only in the edited attribute.

Rates (cfg): p_logic = 1.0 of logic-precond rows, p_browser = 0.15 of eligible short browser rows (choice, target
an interactive element, a "none" option, an acting op; 13.7k of 60k train rows). Measured on the seed-0 stream over
one run's budget (56 steps >= 32,768 tokens): 1,363 questions, of which 79 pairs (31 logic-precond, 48 browser) =
158 pair members (11.6 % of questions); the 79 twins take 57k of 2.12M tokens (2.7 %); total tokens 2.117M vs
2.133M on main (the steps' fill varies with which examples land in them).
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from zlab.data import TRANSFORMS, Transform, category

ACTION_Q = "Which page element should be acted on next to carry out the instruction?"

# ------------------------------------------------------------------------------------------- logic-precond rows
STATE_WORDS = {"input": ("empty", "filled in"), "checkbox": ("unchecked", "checked"),
               "dropdown": ("nothing selected", "an option selected")}
SOLD_OUT = " (no options available: sold out)"
_FIELD = re.compile(r'^(input|checkbox|dropdown) "([^"]+)" — (empty|filled in|unchecked|checked|nothing selected|an option selected)(.*)$')
_GOAL = re.compile(r'^button "([^"]+)" — disabled until (.+)$')
_COND = re.compile(r'^(?:(.+) is filled in|(.+) is checked|an option is selected in (.+))$')


def _conds(text: str) -> list[str]:
    out = []
    for part in text.split(" and "):
        m = _COND.match(part.strip())
        if not m:
            return []
        out.append(next(g for g in m.groups() if g))
    return out


def parse_precond(ex: dict) -> dict | None:
    """-> {goal, goal_key, fields: {name: {key, kind, done, sold_out, needs}}, needs: {node: [names]}} or None."""
    if ex.get("source") != "structured-logic" or (ex.get("meta") or {}).get("task") != "logic-precond":
        return None
    crit = ex["question"].get("criteria")
    if not isinstance(crit, dict) or "none" not in crit:
        return None
    goal = goal_key = None
    fields: dict[str, dict] = {}
    needs: dict[str, list[str]] = {}
    for key, text in crit.items():
        m = _GOAL.match(text or "")
        if m:
            goal, goal_key = m.group(1), key
            needs[goal] = _conds(m.group(2))
            continue
        m = _FIELD.match(text or "")
        if not m:
            continue
        kind, name, state, rest = m.groups()
        sold = rest.endswith(SOLD_OUT)
        rest = rest[: -len(SOLD_OUT)] if sold else rest
        fields[name] = {"key": key, "kind": kind, "done": state == STATE_WORDS[kind][1], "sold_out": sold}
        if rest.startswith("; usable once "):
            needs[name] = _conds(rest[len("; usable once "):])
    if goal is None or not needs.get(goal):
        return None
    if any(n not in fields for ns in needs.values() for n in ns):
        return None
    return {"goal": goal, "goal_key": goal_key, "fields": fields, "needs": needs}


def solve_precond(g: dict, done: dict[str, bool]) -> str:
    """build_logic.py's solver: walk from the goal to the first undone requirement whose own requirements are met.
    -> the option key to act on, or "none" when that element is sold out."""
    node, seen = g["goal"], set()
    while True:
        missing = [n for n in g["needs"].get(node, []) if not done[n]]
        if not missing or node in seen:
            break
        seen.add(node)
        node = missing[0]
    if node == g["goal"]:
        return g["goal_key"]
    return "none" if g["fields"][node]["sold_out"] else g["fields"][node]["key"]


def precond_twin(ex: dict, rng: random.Random) -> dict | None:
    g = parse_precond(ex)
    if g is None:
        return None
    done = {n: f["done"] for n, f in g["fields"].items()}
    label = solve_precond(g, done)
    if label != str(ex["label"]):
        return None  # our parse does not reproduce the generator's label: leave the row alone
    unmeet, other = [], []
    for name, f in g["fields"].items():
        flipped = {**done, name: not done[name]}
        new = solve_precond(g, flipped)
        if new != label:
            (unmeet if done[name] else other).append((name, new))
    pick = unmeet if unmeet and (not other or rng.random() < 0.75) else other
    if not pick:
        return None
    name, new_label = rng.choice(sorted(pick))
    f = g["fields"][name]
    old_word, new_word = (STATE_WORDS[f["kind"]][1], STATE_WORDS[f["kind"]][0]) if f["done"] else STATE_WORDS[f["kind"]]
    crit = dict(ex["question"]["criteria"])
    prefix = f'{f["kind"]} "{name}" — '
    text = crit[f["key"]]
    assert text.startswith(prefix + old_word)
    crit[f["key"]] = prefix + new_word + text[len(prefix + old_word):]
    return {**ex, "question": {**ex["question"], "criteria": crit}, "label": new_label,
            "id": f'{ex.get("id", "")}-edit', "meta": {**(ex.get("meta") or {}), "edit_pair": "precond", "edited": f["key"]}}


# ------------------------------------------------------------------------------------------------ browser rows
TARGET_ROLES = {"button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio", "tab", "menuitem", "option",
                "spinbutton", "listbox", "switch"}
BLOCKER = {"textbox": ("filled in", "empty", "{f} is filled in"), "searchbox": ("filled in", "empty", "{f} is filled in"),
           "spinbutton": ("filled in", "empty", "{f} is filled in"), "checkbox": ("checked", "unchecked", "{f} is checked"),
           "combobox": ("an option selected", "nothing selected", "an option is selected in {f}")}
ACTING_OPS = {"CLICK", "TYPE", "SELECT", "click", "fill", "select", "type", "check"}
_NAME = re.compile(r'^(\S+)(?: level \d)? ("(?:[^"\\]|\\.)*")')


def _role_name(text: str) -> tuple[str, str] | None:
    m = _NAME.match(text or "")
    if not m:
        return None
    try:
        return m.group(1), json.loads(m.group(2))
    except json.JSONDecodeError:
        return None


def browser_eligible(ex: dict) -> bool:
    if category(ex) != "short" or ex["question"].get("type", "choice") != "choice":
        return False
    crit, label = ex["question"].get("criteria"), str(ex["label"])
    if not isinstance(crit, dict) or "none" not in crit or label == "none" or label not in crit:
        return False
    meta = ex.get("meta") or {}
    if (meta.get("op") or meta.get("action")) not in ACTING_OPS:
        return False
    rn = _role_name(crit[label])
    return rn is not None and rn[0] in TARGET_ROLES


def browser_pair(ex: dict, rng: random.Random) -> tuple[dict, dict] | None:
    """-> (original re-asked with the precondition met, twin with it unmet)."""
    if not browser_eligible(ex):
        return None
    crit, label = dict(ex["question"]["criteria"]), str(ex["label"])
    target = crit[label]
    names = {}
    for k, t in crit.items():
        rn = _role_name(t) if k not in ("none", label) else None
        if rn and rn[1].strip():
            names.setdefault(rn[1], []).append((k, rn[0]))
    blockers = [(k, role, name) for name, ks in names.items() for k, role in ks
                if len(ks) == 1 and role in BLOCKER and '"' not in name and len(name) <= 60]
    q2 = {**ex["question"], "instructions": ACTION_Q}
    meta = {**(ex.get("meta") or {})}
    if blockers and rng.random() < 0.8:
        k, role, name = rng.choice(sorted(blockers))
        met, unmet, cond = BLOCKER[role]
        a = dict(crit)
        a[label] = f"{target} — disabled until {cond.format(f=name)}"
        b = dict(a)
        a[k] = f"{crit[k]} — {met}"
        b[k] = f"{crit[k]} — {unmet}"
        orig = {**ex, "question": {**q2, "criteria": a}, "id": f'{ex.get("id", "")}-pre',
                "meta": {**meta, "edit_pair": "browser-blocker", "edited": k}}
        twin = {**ex, "question": {**q2, "criteria": b}, "label": k, "id": f'{ex.get("id", "")}-edit',
                "meta": {**meta, "edit_pair": "browser-blocker", "edited": k}}
        return orig, twin
    word = "read-only" if (meta.get("op") or meta.get("action")) in ("TYPE", "fill", "type") else "disabled"
    on = "editable" if word == "read-only" else "enabled"
    a, b = dict(crit), dict(crit)
    a[label] = f"{target} — {on}"
    b[label] = f"{target} — {word}"
    orig = {**ex, "question": {**q2, "criteria": a}, "id": f'{ex.get("id", "")}-pre',
            "meta": {**meta, "edit_pair": "browser-disabled", "edited": label}}
    twin = {**ex, "question": {**q2, "criteria": b}, "label": "none", "id": f'{ex.get("id", "")}-edit',
            "meta": {**meta, "edit_pair": "browser-disabled", "edited": label}}
    return orig, twin


# ------------------------------------------------------------------------------------------------- transform

@TRANSFORMS.register("edit_pairs")
class EditPairs(Transform):
    """Stateful over the (sequential, deterministic) training stream: a chosen example is returned now and its twin
    replaces the next example. Never touches eval (train=False returns ex unchanged)."""

    def __init__(self, p_logic: float = 1.0, p_browser: float = 0.15, **cfg: Any) -> None:
        super().__init__(p_logic=p_logic, p_browser=p_browser, **cfg)
        self.p_logic, self.p_browser = float(p_logic), float(p_browser)
        self.pending: dict | None = None
        self.stats = {"seen": 0, "pairs_precond": 0, "pairs_browser": 0, "replaced": 0}

    def __call__(self, ex, rng, train):
        if not train:
            return ex
        self.stats["seen"] += 1
        if self.pending is not None:
            twin, self.pending = self.pending, None
            self.stats["replaced"] += 1
            return twin
        u = rng.random()
        if parse_precond(ex) is not None and u < self.p_logic:
            twin = precond_twin(ex, rng)
            if twin is not None:
                if "_order" in ex:
                    twin["_order"] = list(ex["_order"])
                self.pending = twin
                self.stats["pairs_precond"] += 1
            return ex
        if u < self.p_browser:
            pair = browser_pair(ex, rng)
            if pair is not None:
                orig, twin = pair
                if "_order" in ex:
                    orig["_order"], twin["_order"] = list(ex["_order"]), list(ex["_order"])
                self.pending = twin
                self.stats["pairs_browser"] += 1
                return orig
        return ex
