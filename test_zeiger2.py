"""Zeiger2 check against a real checkpoint: python test_zeiger2.py models/zeiger2-3.5-0.8b

Engine() must pick the Zeiger2 engine from the checkpoint's config and answer every question type in the same
format as Zeiger 0.6B. Needs transformers >= 5 (requirements-zeiger2.txt).
"""

from __future__ import annotations

import sys

from zeiger import Engine, Zeiger2Engine

engine = Engine(sys.argv[1] if len(sys.argv) > 1 else "models/zeiger2-3.5-0.8b")
assert isinstance(engine, Zeiger2Engine), type(engine)
state = {"instruction": 'Click the "Sign in" button', "page": {"title": "Shop", "url": "https://shop.test/"}}

answer = engine.decide(state, {"q": {"type": "choice", "instructions": "Which page element does the instruction refer to?",
                                     "criteria": {"e0": 'link "Home" href=/', "e1": 'button "Sign in" in header',
                                                  "e2": 'input "Search"', "none": "none of the listed elements fits this step"}}})["q"]
assert answer["choice"] == "e1" and list(answer["probabilities"]) == ["e0", "e1", "e2", "none"], answer
assert abs(sum(answer["probabilities"].values()) - 1) < 1e-3

yes_no = engine.decide(state, {"q": {"type": "noul", "instructions": "Does the page have a sign-in button?",
                                     "criteria": {"false": "no", "true": "yes"}}})["q"]
assert yes_no["type"] == "noul" and 0.0 <= yes_no["noul"] <= 1.0

score = engine.decide(state, {"q": {"type": "score", "instructions": "How relevant is the page to buying shoes?",
                                    "criteria": ["not at all", "a little", "very"]}})["q"]
assert score["type"] == "score" and score["score"] in (0, 1, 2) and len(score["probabilities"]) == 3

print("zeiger2 tests passed", engine.info()["arch"])
