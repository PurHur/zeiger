"""Smoke tests: python test_zeiger.py (CPU, tiny random backbone, no checkpoint needed)."""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer, Qwen3Config

from zeiger import Engine, Zeiger
from zeiger.model import ATTENTION, BASE_MODEL, register_attention


def tiny_engine() -> Engine:
    register_attention()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    engine = Engine.__new__(Engine)
    engine.model = Zeiger(AutoModel.from_config(config, attn_implementation=ATTENTION), 2).eval()
    engine.tokenizer, engine.config = tokenizer, {"arch": "qwen3-marker"}
    engine.device, engine.dtype = torch.device("cpu"), torch.float32
    engine.max_len, engine.chunk_tokens, engine.token_budget = 131072, 1024, 8192
    engine.temperatures, engine.stats = {}, {"questions": 0, "batches": 0, "ms": 0.0}
    return engine


def choice(n: int) -> dict:
    criteria = {f"e{i}": f'link "Article {i}" href=/wiki/Item_{i} in main' for i in range(n)}
    criteria["none"] = "none of the listed elements fits this step"
    return {"type": "choice", "instructions": "Which page element does the instruction refer to?", "criteria": criteria}


engine = tiny_engine()
state = {"instruction": 'Click "Article 3"', "page": {"title": "T", "url": "https://x.test/"}}

answer = engine.decide(state, {"q": choice(12)})["q"]
assert set(answer["probabilities"]) == set(choice(12)["criteria"])
assert abs(sum(answer["probabilities"].values()) - 1) < 1e-3

whole_page = engine.decide(state, {"q": choice(900)})["q"]
assert len(whole_page["probabilities"]) == 901, "a chunked page lost options"

mixed = {f"q{i}": choice(n) for i, n in enumerate([5, 300, 9, 120, 40])}
answers = engine.decide(state, mixed)
for qid, question in mixed.items():
    assert set(answers[qid]["probabilities"]) == set(question["criteria"]), f"{qid} answered with other options"

yes_no = engine.decide(state, {"q": {"type": "noul", "instructions": "Is there a basket?",
                                     "criteria": {"false": "no", "true": "yes"}}})["q"]
assert yes_no["type"] == "noul" and 0.0 <= yes_no["noul"] <= 1.0

score = engine.decide(state, {"q": {"type": "score", "instructions": "How relevant?",
                                    "criteria": ["not at all", "a little", "very"]}})["q"]
assert score["type"] == "score" and score["score"] in (0, 1, 2)

print("zeiger smoke tests passed")
