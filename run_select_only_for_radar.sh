#!/usr/bin/env bash
# 对已有的 pretrain 目录跑 select_only_stats，获取真实 trace 数据，重新生成雷达图
# 用法: ./run_select_only_for_radar.sh <pretrain_dir>

set -e
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
    echo "用法: $0 <pretrain_dir>"
    echo "示例: $0 pretrain_results/2026-01-25_01-04-17_gsm8k_id16+id51+id21+id11+id56_topL3_plan3_n600"
    exit 1
fi

PRETRAIN_DIR="$1"
if [ ! -d "$PRETRAIN_DIR" ]; then
    echo "错误: 目录不存在: $PRETRAIN_DIR"
    exit 1
fi

# 从目录名推断参数
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
    TASK_POOL="runtime/balanced_task_pool_1_2001.jsonl"
    BBH_TYPES=""
else
    echo "警告: 无法从目录名推断 benchmark，使用默认值"
    BENCHMARK=""
    TASK_POOL="runtime/balanced_task_pool_1_2001.jsonl"
    BBH_TYPES=""
fi

# 从目录名提取参数（假设格式: YYYY-MM-DD_HH-MM-SS_<benchmark>_<agent_tag>_topL<topL>_plan<plan_k>_n<n>）
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

# Agents: 从目录名或默认
AGENTS="16,51,21,11,56"

# Phase sizes: 从 accuracy_summary.csv 推断或默认
COLD_N=200
PRETRAIN_N=300
TEST_N=100

echo "=== 运行 select_only_stats ==="
echo "目录: $PRETRAIN_DIR"
echo "Task pool: $TASK_POOL"
echo "Benchmark: ${BENCHMARK:-all}"
echo "Agents: $AGENTS"
echo "topL: $TOPL, plan-k: $PLAN_K"
echo ""

CMD="python3 select_only_stats.py"
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
CMD="$CMD --runtime-dir runtime"
CMD="$CMD --agents \"$AGENTS\""
CMD="$CMD --topL $TOPL"
CMD="$CMD --plan-k $PLAN_K"
CMD="$CMD --cot-count 3"
CMD="$CMD --test-only"
CMD="$CMD --outdir \"$PRETRAIN_DIR\""

echo "执行: $CMD"
eval $CMD

echo ""
echo "=== 重新生成雷达图 ==="
python3 plot_from_json.py "$PRETRAIN_DIR"

echo ""
echo "✅ 完成！真实 trace 数据已写入 $PRETRAIN_DIR"
