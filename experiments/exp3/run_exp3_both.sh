#!/bin/bash
set -e
# ==========================================================
# Exp3 Launcher (Simulation)
# Runs both Shock A and Shock B sequentially
# ==========================================================

echo "=========================================="
echo "Exp3 (Simulation): Running Shock A & B"
echo "=========================================="
echo ""

PYTHON_RUNNER="experiments/exp3/sim/exp3_sim.py"
BASE_DIR="experiments/exp3/sim/results/sim_exp3_robustness"

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
echo "✅ Exp3 Simulation Completed"
echo "=========================================="
echo ""
echo "Results saved under:"
echo "  $BASE_DIR/"
echo ""
echo "Structure:"
echo "  ├── ShockA/YYYY-MM-DD_HH-MM-SS/"
echo "  └── ShockB/YYYY-MM-DD_HH-MM-SS/"
echo ""
