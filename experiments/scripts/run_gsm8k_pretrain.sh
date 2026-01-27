#!/usr/bin/env bash
# GSM8K pretrain + radar chart
# Agents: 16 (deepseek-v3), 51 (x-ai-grok), 21 (gpt-oss-120b), 11 (deepseek-v3-0324), 56 (gemini-2.5-flash-lite)
# Task pool: exp5 = gsm8k_full. If less than 600, pretrain will auto-reduce test-n.

set -e

# Get the project root directory (two levels up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

TASK_POOL="${TASK_POOL:-symphony-data-generator/data/exp5/task_pool.jsonl}"

python3 "$EXPERIMENTS_DIR/pretrain.py" \
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
  --runtime-dir "$EXPERIMENTS_DIR/configs" \
  --plot-acc \
  --print-each-step \
  --plot-radar
