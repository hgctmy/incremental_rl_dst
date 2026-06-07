# Incremental End-to-End Spoken Dialogue State Tracking with a Multimodal LLM and Reinforcement Learning

Code for our Interspeech 2026 paper.

> **Incremental End-to-End Spoken Dialogue State Tracking with a Multimodal LLM and Reinforcement Learning**
> Tomoya Higuchi
> *Interspeech 2026*

## Overview

We train [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) to perform **incremental Dialogue State Tracking (DST)** directly from audio — without a separate ASR step.

Instead of outputting the full belief state at every turn, the model outputs **diff operations** (set / update / delete) relative to the previous state:

```
<transcript>
System: do you have a price preference .
User: i'd like something cheap .
</transcript>
<answer>set(restaurant.pricerange=cheap)</answer>
```

Training uses a two-stage pipeline:
1. **SFT** — supervised fine-tuning with QLoRA (rank 64, α 128, 4-bit NF4)
2. **GRPO** — reinforcement learning with a composite reward (transcript WER + diff F1 + exact match + format)

Evaluated on [SpokenWOZ](https://github.com/ZekangLi/SpokenWOZ).

## Repository structure

```
.
├── prompts/
│   ├── incremental.txt       # system prompt for incremental DST
│   └── fullstate.txt         # system prompt for full-state DST (baseline)
├── scripts/
│   ├── train/
│   │   ├── train_sft.sh                          # Stage 1: SFT
│   │   ├── train_grpo.sh                         # Stage 2: GRPO (main)
│   │   ├── train_grpo_ablation_no_transcript.sh  # ablation: no WER reward
│   │   ├── train_sft_fullstate.sh                # full-state baseline SFT
│   │   ├── train_grpo_fullstate.sh               # full-state baseline GRPO
│   │   ├── split_audio.py            # extract per-sample user-turn audio from SpokenWOZ WAVs
│   │   ├── prepare_data.py           # convert SpokenWOZ raw data → GRPO JSONL
│   │   ├── convert_to_sft.py         # convert GRPO-format data → SFT format
│   │   ├── prepare_fullstate_data.py # convert incremental data → full-state format
│   │   └── sample_val.py             # sample a small validation subset
│   ├── infer/
│   │   ├── infer.py / infer_fullstate.py         # vLLM batch inference
│   │   ├── infer_oracle.sh / infer_predicted.sh  # oracle / cascading inference
│   │   └── infer_fullstate_oracle.sh / infer_fullstate_predicted.sh
│   └── eval/
│       ├── eval.py            # compute WER / JGA / Slot F1 (incremental)
│       ├── eval_fullstate.py  # compute WER / JGA / Slot F1 (full-state)
│       └── plot_jga_by_turn.py
├── src/
│   └── reward.py   # ms-swift ORM plugin: GRPO reward functions
├── tests/
└── pyproject.toml
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

GPU-specific packages (`flash-attn`, `vllm`, etc.) are listed in `pyproject.toml` and resolved by `uv sync` for the target environment.

## Data

Download SpokenWOZ from [spokenwoz.github.io](https://spokenwoz.github.io/) and place the raw files under `data/raw/`:

```
data/raw/
  train.json
  val.json
  test.json
  audio/          # WAV files referenced by each JSON
```

Each model input is the text dialogue history plus the audio of the latest user
turn. SpokenWOZ ships one WAV per dialogue, so first extract per-sample
user-turn audio (named `{dialogue_id}_{sys_idx}_{user_idx}.wav`) using the
word-level timing in the JSON:

```bash
python scripts/train/split_audio.py --data data/raw/train.json --audio-dir data/raw/audio --output-dir data/audio/train
python scripts/train/split_audio.py --data data/raw/val.json   --audio-dir data/raw/audio --output-dir data/audio/val
python scripts/train/split_audio.py --data data/raw/test.json  --audio-dir data/raw/audio --output-dir data/audio/test
```

Then convert to GRPO JSONL format. Each sample carries the diff-operation target
(set / update / delete) derived from the belief state on the *next* system turn,
plus the previous state from the current system turn:

```bash
python scripts/train/prepare_data.py --data data/raw/train.json --output data/train.jsonl
python scripts/train/prepare_data.py --data data/raw/val.json   --output data/val.jsonl
python scripts/train/prepare_data.py --data data/raw/test.json  --output data/test.jsonl
```

Point training/inference at the split audio with `--audio-base-dir data/audio/<split>`.

Full-state baseline data is auto-generated from the incremental data by the training scripts.

## Training

### Stage 1 — SFT

```bash
bash scripts/train/train_sft.sh
```

Converts GRPO-format data to SFT format automatically, then runs QLoRA fine-tuning.

### Stage 2 — GRPO

```bash
# Start from SFT checkpoint (set SFT_CHECKPOINT)
SFT_CHECKPOINT=output/sft_incremental_dst/checkpoint-xxxx \
bash scripts/train/train_grpo.sh
```

Uses the custom reward plugin at `src/reward.py` via ms-swift's `--external_plugins` interface.

Default: 8 GPUs (6 for training, 2 for vLLM inference during rollout).

### Ablation (no transcript reward)

```bash
bash scripts/train/train_grpo_ablation_no_transcript.sh
```

## Inference

```bash
# Oracle mode (ground-truth dialogue history)
ADAPTER=output/grpo_incremental_dst/checkpoint-xxxx \
bash scripts/infer/infer_oracle.sh

# Predicted mode (model's own transcript cascades across turns)
ADAPTER=output/grpo_incremental_dst/checkpoint-xxxx \
bash scripts/infer/infer_predicted.sh
```

Both scripts run inference then evaluation automatically and write results to `output/`.

## Evaluation

```bash
python scripts/eval/eval.py \
    --input output/vllm_inference_results/oracle/predictions.jsonl \
    --output output/vllm_inference_results/oracle/metrics.json
```

Metrics: transcript WER, Joint Goal Accuracy (JGA), Slot F1.