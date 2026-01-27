#!/usr/bin/env bash
# Balanced task pool (混合 benchmark) pretrain + 雷达图一键跑
# Task pool: runtime/balanced_task_pool_1_2001.jsonl，不按 benchmark 过滤，n=600 随机抽样
# Agents: 16 (deepseek-v3), 51 (x-ai-grok), 21 (gpt-oss-120b), 11 (deepseek-v3-0324), 56 (gemini-2.5-flash-lite)

set -e
cd "$(dirname "$0")"

TASK_POOL="${TASK_POOL:-runtime/balanced_task_pool_1_2001.jsonl}"

python3 Pre-train.py \
  --task-pool "$TASK_POOL" \
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
