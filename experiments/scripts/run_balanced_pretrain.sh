#!/usr/bin/env bash
# Balanced task pool (mixed benchmark) pretrain + radar chart
# Task pool: configs/balanced_task_pool_1_2001.jsonl, no benchmark filter, n=600 random sample
# Agents: 16 (deepseek-v3), 51 (x-ai-grok), 21 (gpt-oss-120b), 11 (deepseek-v3-0324), 56 (gemini-2.5-flash-lite)

set -e

# Get the project root directory (two levels up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

TASK_POOL="${TASK_POOL:-$EXPERIMENTS_DIR/configs/balanced_task_pool_1_2001.jsonl}"

python3 "$EXPERIMENTS_DIR/pretrain.py" \
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
  --runtime-dir "$EXPERIMENTS_DIR/configs" \
  --plot-acc \
  --print-each-step \
  --plot-radar
