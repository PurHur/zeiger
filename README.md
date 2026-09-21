# Zeiger

A small decision model that reads a **whole web page** and points at the element an instruction means — or at
**"none of these"** — in one forward pass. This repository is the inference engine; bring your own checkpoint.

```python
from zeiger import Engine

e = Engine("models/zeiger-0.6b")                 # device="auto" -> GPU when ROCm/CUDA is present
answers = e.decide(
    {"instruction": 'Click the "Sign in" button', "page": {"title": "Shop", "url": "https://shop.test/"}},
    {"q": {"type": "choice",
           "instructions": "Which page element does the instruction refer to?",
           "criteria": {"e0": 'link "Home" href=/',
                        "e1": 'button "Sign in" in header',
                        "e2": 'input "Search"',
                        "none": "none of the listed elements fits this step"}}})

answers["q"]["choice"]         # 'e1'
answers["q"]["confidence"]     # 0.986
answers["q"]["probabilities"]  # every option, calibrated
```

* 40 to 1,500 options per question, scored against each other rather than one at a time
* `"none"` is a real option, so "not on this page" is an answer
* pages beyond the window are encoded in 2k-token chunks: attention cost stays linear in page size
* confidences are calibrated by a temperature stored in the checkpoint

## Install

```bash
python -m venv .venv && . .venv/bin/activate
# AMD (ROCm):
pip install --index-url https://download.pytorch.org/whl/rocm7.0 torch==2.10.0
# NVIDIA or CPU: the matching torch wheel from pytorch.org
pip install -r requirements.txt          # transformers is pinned <5: see requirements.txt
```

On ROCm, start any process with:

```bash
export LD_PRELOAD=/opt/rocm-7.2.3/lib/libhsa-runtime64.so.1     # the wheel's bundled HSA runtime segfaults on gfx1151
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1                # enables the flash attention kernels
```

## Serve

```bash
python serve.py --model models/zeiger-0.6b --port 8173          # POST /decide, GET / for status
```

Or with Docker — put a checkpoint in `./models/zeiger-0.6b` first:

```bash
docker compose up zeiger                        # CPU, portable
docker compose --profile rocm up zeiger-rocm    # AMD GPU
```

The ROCm service passes `/dev/kfd` and `/dev/dri` through and mounts the **host's** ROCm runtime read-only: the
rocm7.0 wheel ships an HSA runtime that segfaults on gfx1151, so `LD_PRELOAD` points at the host's newer one.
Adjust that path in `docker-compose.yml` to match your installation.

```bash
curl -s localhost:8173/decide -H 'content-type: application/json' -d '{
  "state": {"instruction": "Open the basket"},
  "questions": {"q": {"type": "choice", "instructions": "Which element?",
                      "criteria": {"a": "link Home", "b": "button Basket (3)", "none": "none of these"}}}}'
```

```json
{"answers": {"q": {"type": "choice", "choice": "b", "confidence": 0.973,
                   "probabilities": {"a": 0.0002, "b": 0.973, "none": 0.027}}}, "ms": 1592.8}
```

Question types: `choice` (pick one option, `"none"` allowed), `noul` (yes/no — the answer is the probability the
statement holds) and `score` (ordinal levels). A question whose options do not fit the window returns
`{"error": "options do not fit the window"}` instead of a guess.

## Model

`Qwen3-0.6B-Base` (Apache-2.0, 28 layers) with its language-model head removed: the transformer body is an
encoder and nothing is generated. The page becomes one sequence,

```
<state as JSON> \n choice question: <instruction> \n opt0 <M> opt1 <M> … optK-1 <M> <eos>
```

where `<M>` is a token the Qwen tokenizer already reserves. The hidden state at each marker is that option's
summary; ~18M parameters of head then read the markers — a 2-layer bidirectional transformer across the options,
a question-type embedding, and a scorer giving one logit per option. The head costs O(K²) in the number of
options, never O(L²) in page length.

## Hardware

Tuned for AMD Strix Halo (gfx1151): a mask-free causal attention kernel keeps SDPA on its flash path, bf16 on
GPU, and micro-batches binned by chunk length so short questions do not pay for long ones.

PyTorch has no Vulkan inference backend, and llama.cpp cannot run this architecture, so the choice on AMD is
ROCm or CPU. Measure both:

```bash
python bench.py --model models/zeiger-0.6b --devices cuda,cpu
```

## Checkpoints

The engine loads any export directory with this layout:

```
models/zeiger-0.6b/
  model.safetensors        # backbone + head
  rl_agent_config.json     # arch, chunk_tokens, max_len, temperature_by_options
  encoder/                 # the backbone's config
  tokenizer/
```

`rl_agent_config.json` carries the window, chunk size and calibration temperatures, so a checkpoint brings its
own serving settings.

## Tests

```bash
python test_zeiger.py       # CPU, tiny random backbone, no checkpoint needed
```

## Licence

Code: Apache-2.0. A checkpoint carries its own terms, depending on what it was trained on.
