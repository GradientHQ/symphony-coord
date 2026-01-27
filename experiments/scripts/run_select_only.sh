#!/usr/bin/env bash
# Run select_only_stats on an existing pretrain directory to get real trace data and regenerate radar charts
# Usage: ./run_select_only.sh <pretrain_dir>

set -e

# Get the project root directory (two levels up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pretrain_dir>"
    echo "Example: $0 pretrain_results/2026-01-25_01-04-17_gsm8k_id16+id51+id21+id11+id56_topL3_plan3_n600"
    exit 1
fi

PRETRAIN_DIR="$1"
if [ ! -d "$PRETRAIN_DIR" ]; then
    echo "Error: Directory does not exist: $PRETRAIN_DIR"
    exit 1
fi

# Infer parameters from directory name
DIRNAME=$(basename "$PRETRAIN_DIR")
if [[ "$DIRNAME" == *"gsm8k"* ]]; then
    BENCHMARK="gsm8k"
    TASK_POOL="symphony-data-generator/data/exp5/task_pool.jsonl"
    BBH_TYPES=""
elif [[ "$DIRNAME" == *"bbh"* ]]; then
    BENCHMARK="bbh"
    TASK_POOL="symphony-data-generator/data/benchmarks/full/bbh_full.jsonl"
    BBH_TYPES="sports_understanding,movie_recommendation,boolean_expressions"
elif [[ "$DIRNAME" == *"all"* ]]; then
    BENCHMARK=""
    TASK_POOL="$EXPERIMENTS_DIR/configs/balanced_task_pool_1_2001.jsonl"
    BBH_TYPES=""
else
    echo "Warning: Cannot infer benchmark from directory name, using defaults"
    BENCHMARK=""
    TASK_POOL="$EXPERIMENTS_DIR/configs/balanced_task_pool_1_2001.jsonl"
    BBH_TYPES=""
fi

# Extract parameters from directory name (format: YYYY-MM-DD_HH-MM-SS_<benchmark>_<agent_tag>_topL<topL>_plan<plan_k>_n<n>)
if [[ "$DIRNAME" =~ topL([0-9]+) ]]; then
    TOPL="${BASH_REMATCH[1]}"
else
    TOPL=3
fi

if [[ "$DIRNAME" =~ plan([0-9]+) ]]; then
    PLAN_K="${BASH_REMATCH[1]}"
else
    PLAN_K=3
fi

# Agents: from directory name or default
AGENTS="16,51,21,11,56"

# Phase sizes: from accuracy_summary.csv or default
COLD_N=200
PRETRAIN_N=300
TEST_N=100

echo "=== Running select_only_stats ==="
echo "Directory: $PRETRAIN_DIR"
echo "Task pool: $TASK_POOL"
echo "Benchmark: ${BENCHMARK:-all}"
echo "Agents: $AGENTS"
echo "topL: $TOPL, plan-k: $PLAN_K"
echo ""

CMD="python3 $SCRIPTS_DIR/analysis/select_only_stats.py"
CMD="$CMD --task-pool \"$TASK_POOL\""
if [ -n "$BENCHMARK" ]; then
    CMD="$CMD --benchmark $BENCHMARK"
fi
if [ -n "$BBH_TYPES" ]; then
    CMD="$CMD --bbh-task-types \"$BBH_TYPES\""
fi
CMD="$CMD --n 600"
CMD="$CMD --cold-n $COLD_N"
CMD="$CMD --pretrain-n $PRETRAIN_N"
CMD="$CMD --test-n $TEST_N"
CMD="$CMD --seed 42"
CMD="$CMD --runtime-dir \"$EXPERIMENTS_DIR/configs\""
CMD="$CMD --agents \"$AGENTS\""
CMD="$CMD --topL $TOPL"
CMD="$CMD --plan-k $PLAN_K"
CMD="$CMD --cot-count 3"
CMD="$CMD --test-only"
CMD="$CMD --outdir \"$PRETRAIN_DIR\""

echo "Executing: $CMD"
eval $CMD

echo ""
echo "=== Regenerating radar charts ==="
python3 "$SCRIPTS_DIR/plotting/routing/plot_from_json.py" "$PRETRAIN_DIR"

echo ""
echo "Done! Real trace data has been written to $PRETRAIN_DIR"
