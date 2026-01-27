#!/usr/bin/env bash
# BBH pretrain + 雷达图一键跑（仅 3 个 task type）
# Agents: 16 (deepseek-v3), 51 (x-ai-grok), 21 (gpt-oss-120b), 11 (deepseek-v3-0324), 56 (gemini-2.5-flash-lite)
# Task types: sports_understanding, movie_recommendation, boolean_expressions

set -e
cd "$(dirname "$0")"

TASK_POOL="${TASK_POOL:-symphony-data-generator/data/benchmarks/full/bbh_full.jsonl}"

python3 Pre-train.py \
  --task-pool "$TASK_POOL" \
  --benchmark bbh \
  --bbh-task-types "sports_understanding,movie_recommendation,boolean_expressions" \
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
