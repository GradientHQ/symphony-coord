# Experiment 3: System Performance Optimization

This experiment evaluates LinUCB's ability to optimize routing based on system-level metrics (latency and load).

## Overview

**Research Question**: Can LinUCB optimize routing based on latency and load, not just task-agent matching?

**Test Scenarios**:
1. `latency_heterogeneous`: Agents have different response latencies
2. `load_burst`: Dynamic load spikes on specific agents
3. `combined`: Both latency differences and load bursts
4. `baseline`: Normal conditions (control)

## Directory Structure

```
exp3/
├── sim_system_optimization.py  # Main simulation
├── plot_results.py             # Visualization
├── configs/
│   └── scenarios.yaml          # Scenario definitions
├── run_exp3.sh                 # Quick runner
└── README.md
```

## Running the Experiment

### Quick Start

```bash
bash experiments/exp3/run_exp3.sh
```

### Manual Execution

```bash
# Step 1: Run simulation
python3 experiments/exp3/sim_system_optimization.py \
  --config experiments/exp3/configs/scenarios.yaml

# Step 2: Generate plots
python3 experiments/exp3/plot_results.py \
  --result-dir experiments/exp3/results
```

## Scenarios

| Scenario | Description |
|----------|-------------|
| `latency_heterogeneous` | Agents A-E have latency multipliers: 2.5x, 0.5x, 1.0x, 0.7x, 1.8x |
| `load_burst` | Agent A overloaded t=200-400, Agent B overloaded t=500-700 |
| `combined` | Both latency differences and Agent B burst at t=300-500 |
| `baseline` | Normal conditions |

## Compared Strategies

| Strategy | Description | Learns? |
|----------|-------------|---------|
| `always_A` | Always select strongest agent A | No |
| `static_rule` | Hardcoded: simple→B, hard→A | No |
| `random` | Random selection from TopL | No |
| `linucb` | LinUCB adaptive routing | Yes |

## Key Metrics

| Metric | Description |
|--------|-------------|
| Success Rate | Task completion rate |
| Avg Latency | Mean response time (ms) |
| P95 Latency | 95th percentile latency |
| Latency Efficiency | success_rate / avg_latency × 1000 |

## Expected Results

- LinUCB learns to prefer low-latency agents (B, D over A, E)
- LinUCB adapts to load bursts within ~50 tasks
- Achieves Pareto improvement: similar success rate with much lower latency

## Output

Results saved to `results/`:
- `summary_all_scenarios.csv`: Aggregate metrics per scenario
- `step_logs_all.csv`: Per-task logs
- `plot_*.png`: Visualization figures
