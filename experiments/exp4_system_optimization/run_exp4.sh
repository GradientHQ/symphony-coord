#!/bin/bash
# Quick script to run Exp4: System Performance Optimization

set -e

echo "=========================================="
echo "Exp4: System Performance Optimization"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "📍 Working directory: $PWD"
echo ""

# Step 1: Run simulation
echo "🚀 Step 1: Running simulation..."
python3 experiments/exp4_system_optimization/sim_system_optimization.py \
  --config experiments/exp4_system_optimization/configs/scenarios.yaml

echo ""
echo "✅ Simulation completed!"
echo ""

# Step 2: Generate plots
echo "📊 Step 2: Generating plots..."
python3 experiments/exp4_system_optimization/plot_results.py \
  --result-dir experiments/result/exp4_system_optimization

echo ""
echo "✅ Plots generated!"
echo ""

# Step 3: Show summary
echo "📈 Step 3: Results summary"
echo "=========================================="
echo ""

if [ -f "experiments/result/exp4_system_optimization/summary_all_scenarios.csv" ]; then
    echo "Summary file:"
    head -n 1 experiments/result/exp4_system_optimization/summary_all_scenarios.csv
    echo ""
    echo "LinUCB results:"
    grep "linucb" experiments/result/exp4_system_optimization/summary_all_scenarios.csv | \
        awk -F',' '{printf "  [%-25s] success=%.3f  latency=%6.1fms  efficiency=%.3f\n", $2, $4, $5, $10}'
    echo ""
else
    echo "⚠️  Summary file not found"
fi

echo "=========================================="
echo "🎉 Exp4 completed successfully!"
echo ""
echo "📁 Results location:"
echo "   experiments/result/exp4_system_optimization/"
echo ""
echo "📊 Key plots (4 total):"
echo "   - plot_scenario_comparison.png      (comprehensive comparison)"
echo "   - plot_latency_learning_curve.png   (learning process)"
echo "   - plot_efficiency_comparison.png    (efficiency metrics)"
echo "   - plot_agent_utilization.png        (agent selection)"
echo ""
echo "📖 Documentation:"
echo "   - README.md"
echo "   - RESULTS_SUMMARY.md"
echo "=========================================="

