#!/bin/bash
# Exp3 完整实验运行脚本
# 运行所有 Simulation、Real Execution 和对比图生成

set -e  # 遇到错误立即退出

PROJECT_ROOT="/Users/caohuixi/symphony2.0"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Exp3 完整实验运行脚本"
echo "=========================================="
echo ""

# ============================================
# 1. Simulation 实验
# ============================================
echo "📊 Step 1: Running Simulation Experiments..."
echo ""

# Shock A (A_unavailable)
echo "  Running Shock A (A_unavailable)..."
python3 experiments/exp3/sim/exp3_sim.py \
  --n 1000 \
  --shock A_unavailable \
  --shock-point 500 \
  --seed 42 \
  --outdir experiments/exp3/sim/results/sim_exp3_robustness

echo "  ✓ Shock A Simulation completed"
echo ""

# Shock B (A_degraded)
echo "  Running Shock B (A_degraded)..."
python3 experiments/exp3/sim/exp3_sim.py \
  --n 1000 \
  --shock A_degraded \
  --shock-point 500 \
  --seed 42 \
  --outdir experiments/exp3/sim/results/sim_exp3_robustness

echo "  ✓ Shock B Simulation completed"
echo ""

# ============================================
# 2. Real Execution 实验
# ============================================
echo "🔬 Step 2: Running Real Execution Experiments..."
echo ""

# 检查任务文件是否存在
TASK_FILE="symphony-data-generator/data/exp3/task_pool.jsonl"
if [ ! -f "$TASK_FILE" ]; then
    echo "  ⚠️  Task file not found: $TASK_FILE"
    echo "  Please run: cd symphony-data-generator && python src/quick_start.py"
    exit 1
fi

echo "  Using task file: $TASK_FILE"
echo ""

# Shock A (A_unavailable) - Real
echo "  Running Shock A (A_unavailable) - Real..."
python3 experiments/exp3/real/exp3_real.py \
  --tasks "$TASK_FILE" \
  --n 200 \
  --shock A_unavailable \
  --shock-point 100 \
  --seed 42 \
  --outdir experiments/exp3/real/results/real_exp3_robustness

echo "  ✓ Shock A Real Execution completed"
echo ""

# Shock B (A_degraded) - Real
# 注意：使用 shock-point=400 而不是 500，确保有足够的 post-shock 任务
echo "  Running Shock B (A_degraded) - Real..."
python3 experiments/exp3/real/exp3_real.py \
  --tasks "$TASK_FILE" \
  --n 500 \
  --shock A_degraded \
  --shock-point 400 \
  --seed 42 \
  --outdir experiments/exp3/real/results/real_exp3_robustness

echo "  ✓ Shock B Real Execution completed"
echo ""

# ============================================
# 3. 生成对比图
# ============================================
echo "📈 Step 3: Generating Comparison Plots..."
echo ""

# 获取最新的结果目录
SIM_SHOCKA_DIR=$(ls -td experiments/exp3/sim/results/sim_exp3_robustness/ShockA/*/ | head -1)
SIM_SHOCKB_DIR=$(ls -td experiments/exp3/sim/results/sim_exp3_robustness/ShockB/*/ | head -1)
REAL_SHOCKA_DIR=$(ls -td experiments/exp3/real/results/real_exp3_robustness/ShockA/*/ | head -1)
REAL_SHOCKB_DIR=$(ls -td experiments/exp3/real/results/real_exp3_robustness/ShockB/*/ | head -1)

# 创建输出目录
mkdir -p experiments/exp3/plot

# Shock A 对比图
echo "  Generating Shock A comparison plot..."
python3 experiments/exp3/plot/plot_sim_vs_real.py \
  --sim "$SIM_SHOCKA_DIR/trajectory_linucb.csv" \
  --real "$REAL_SHOCKA_DIR/trajectory_real.json" \
  --shock-point-sim 500 \
  --shock-point-real 100 \
  --shock-type A_unavailable \
  --out experiments/exp3/plot/sim_vs_real_ShockA.png

echo "  ✓ Shock A comparison plot generated"
echo ""

# Shock B 对比图
# 注意：Sim 和 Real 的 shock_point 不同（Sim: 500, Real: 400）
echo "  Generating Shock B comparison plot..."
python3 experiments/exp3/plot/plot_sim_vs_real.py \
  --sim "$SIM_SHOCKB_DIR/trajectory_linucb.csv" \
  --real "$REAL_SHOCKB_DIR/trajectory_real.json" \
  --shock-point-sim 500 \
  --shock-point-real 400 \
  --shock-type A_degraded \
  --out experiments/exp3/plot/sim_vs_real_ShockB.png

echo "  ✓ Shock B comparison plot generated"
echo ""

# ============================================
# 完成
# ============================================
echo "=========================================="
echo "✅ All experiments completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Simulation: experiments/exp3/sim/results/sim_exp3_robustness/"
echo "  - Real Execution: experiments/exp3/real/results/real_exp3_robustness/"
echo "  - Comparison Plots: experiments/exp3/plot/"
echo ""

