#!/usr/bin/env bash
# GSM8K pretrain + 雷达图一键跑
# Agents: 16 (deepseek-v3), 51 (x-ai-grok), 21 (gpt-oss-120b), 11 (deepseek-v3-0324), 56 (gemini-2.5-flash-lite)
# Task pool: exp5 = gsm8k_full。若不足 600，Pre-train 会自动缩减 test-n。

set -e
cd "$(dirname "$0")"

TASK_POOL="${TASK_POOL:-symphony-data-generator/data/exp5/task_pool.jsonl}"

python3 Pre-train.py \
  --task-pool "$TASK_POOL" \
  --benchmark gsm8k \
  --n 600 \
  --cold-n 200 \
  --pretrain-n 300 \
  --test-n 100 \
  --topL 3 \
  --plan-k 3 \
  --cot-count 3 \
  --ucb-alpha 1 \
  --seed 42 \
  --agents "16,51,21,11,56" \
  --runtime-dir runtime \
  --plot-acc \
  --print-each-step \
  --plot-radar
