#!/bin/bash
set -e
# ==========================================================
# Exp2 Launcher (Simulation)
# Runs both Shock A and Shock B sequentially
# ==========================================================

echo "=========================================="
echo "Exp2 (Simulation): Running Shock A & B"
echo "=========================================="
echo ""

PYTHON_RUNNER="experiments/exp2/sim/exp2_sim.py"
BASE_DIR="experiments/exp2/sim/results/sim_exp2_robustness"

N_TASKS=1000
SHOCK_POINT=500
SEED=42

# -------------------------------
# Shock A
# -------------------------------
echo "[1/2] Running Shock A (A_unavailable)..."

python3 $PYTHON_RUNNER \
  --n $N_TASKS \
  --shock A_unavailable \
  --shock-point $SHOCK_POINT \
  --seed $SEED \
  --outdir $BASE_DIR

echo ""
echo "✔ Shock A completed."
echo ""

# -------------------------------
# Shock B
# -------------------------------
echo "[2/2] Running Shock B (A_degraded)..."

python3 $PYTHON_RUNNER \
  --n $N_TASKS \
  --shock A_degraded \
  --shock-point $SHOCK_POINT \
  --seed $SEED \
  --outdir $BASE_DIR

echo ""
echo "✔ Shock B completed."
echo ""

echo "=========================================="
echo "✅ Exp2 Simulation Completed"
echo "=========================================="
echo ""
echo "Results saved under:"
echo "  $BASE_DIR/"
echo ""
echo "Structure:"
echo "  ├── ShockA/YYYY-MM-DD_HH-MM-SS/"
echo "  └── ShockB/YYYY-MM-DD_HH-MM-SS/"
echo ""
