#!/bin/bash
# Exp2: Complete Experiment Runner
# Runs all Simulation, Real Execution, and comparison plot generation

set -e  # Exit on error

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Exp2 Complete Experiment Runner"
echo "=========================================="
echo ""

# ============================================
# 1. Simulation Experiments
# ============================================
echo "📊 Step 1: Running Simulation Experiments..."
echo ""

# Shock A (A_unavailable)
echo "  Running Shock A (A_unavailable)..."
python3 experiments/exp2/sim/exp2_sim.py \
  --n 1000 \
  --shock A_unavailable \
  --shock-point 500 \
  --seed 42 \
  --outdir experiments/exp2/sim/results/sim_exp2_robustness

echo "  ✓ Shock A Simulation completed"
echo ""

# Shock B (A_degraded)
echo "  Running Shock B (A_degraded)..."
python3 experiments/exp2/sim/exp2_sim.py \
  --n 1000 \
  --shock A_degraded \
  --shock-point 500 \
  --seed 42 \
  --outdir experiments/exp2/sim/results/sim_exp2_robustness

echo "  ✓ Shock B Simulation completed"
echo ""

# ============================================
# 2. Real Execution Experiments
# ============================================
echo "🔬 Step 2: Running Real Execution Experiments..."
echo ""

# Check if task file exists
TASK_FILE="symphony-data-generator/data/exp2/task_pool.jsonl"
if [ ! -f "$TASK_FILE" ]; then
    echo "  ⚠️  Task file not found: $TASK_FILE"
    echo "  Please run: cd symphony-data-generator && python src/quick_start.py"
    exit 1
fi

echo "  Using task file: $TASK_FILE"
echo ""

# Shock A (A_unavailable) - Real
echo "  Running Shock A (A_unavailable) - Real..."
python3 experiments/exp2/real/exp2_real.py \
  --tasks "$TASK_FILE" \
  --n 200 \
  --shock A_unavailable \
  --shock-point 100 \
  --seed 42 \
  --outdir experiments/exp2/real/results/real_exp2_robustness

echo "  ✓ Shock A Real Execution completed"
echo ""

# Shock B (A_degraded) - Real
# Note: Using shock-point=400 instead of 500 to ensure enough post-shock tasks
echo "  Running Shock B (A_degraded) - Real..."
python3 experiments/exp2/real/exp2_real.py \
  --tasks "$TASK_FILE" \
  --n 500 \
  --shock A_degraded \
  --shock-point 400 \
  --seed 42 \
  --outdir experiments/exp2/real/results/real_exp2_robustness

echo "  ✓ Shock B Real Execution completed"
echo ""

# ============================================
# 3. Generate Comparison Plots
# ============================================
echo "📈 Step 3: Generating Comparison Plots..."
echo ""

# Get the latest result directories
SIM_SHOCKA_DIR=$(ls -td experiments/exp2/sim/results/sim_exp2_robustness/ShockA/*/ | head -1)
SIM_SHOCKB_DIR=$(ls -td experiments/exp2/sim/results/sim_exp2_robustness/ShockB/*/ | head -1)
REAL_SHOCKA_DIR=$(ls -td experiments/exp2/real/results/real_exp2_robustness/ShockA/*/ | head -1)
REAL_SHOCKB_DIR=$(ls -td experiments/exp2/real/results/real_exp2_robustness/ShockB/*/ | head -1)

# Create output directory
mkdir -p experiments/exp2/plot

# Shock A comparison plot
echo "  Generating Shock A comparison plot..."
python3 experiments/exp2/plot/plot_sim_vs_real.py \
  --sim "$SIM_SHOCKA_DIR/trajectory_linucb.csv" \
  --real "$REAL_SHOCKA_DIR/trajectory_real.json" \
  --shock-point-sim 500 \
  --shock-point-real 100 \
  --shock-type A_unavailable \
  --out experiments/exp2/plot/sim_vs_real_ShockA.png

echo "  ✓ Shock A comparison plot generated"
echo ""

# Shock B comparison plot
# Note: Sim and Real have different shock_point (Sim: 500, Real: 400)
echo "  Generating Shock B comparison plot..."
python3 experiments/exp2/plot/plot_sim_vs_real.py \
  --sim "$SIM_SHOCKB_DIR/trajectory_linucb.csv" \
  --real "$REAL_SHOCKB_DIR/trajectory_real.json" \
  --shock-point-sim 500 \
  --shock-point-real 400 \
  --shock-type A_degraded \
  --out experiments/exp2/plot/sim_vs_real_ShockB.png

echo "  ✓ Shock B comparison plot generated"
echo ""

# ============================================
# Complete
# ============================================
echo "=========================================="
echo "✅ All experiments completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Simulation: experiments/exp2/sim/results/sim_exp2_robustness/"
echo "  - Real Execution: experiments/exp2/real/results/real_exp2_robustness/"
echo "  - Comparison Plots: experiments/exp2/plot/"
echo ""

