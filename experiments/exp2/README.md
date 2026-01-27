# Experiment 2: Robustness & Recovery under Role Shock

This experiment evaluates system robustness and recovery under non-stationary "role shock" conditions.

## Overview

**Research Question**: Can LinUCB adapt when agent capabilities change unexpectedly?

**Shock Types**:
- `A_unavailable`: Agent A becomes completely unavailable
- `A_degraded`: Agent A's performance degrades significantly

## Directory Structure

```
exp2/
├── sim/
│   └── exp2_sim.py          # Simulation (synthetic tasks)
├── real/
│   ├── exp2_real.py         # Real execution (API calls)
│   └── config_exp2.yaml     # Configuration
├── plot/
│   └── plot_sim_vs_real.py  # Comparison plots
├── scripts/
│   ├── run_all_experiments.sh  # Full pipeline
│   └── run_exp2_both.sh        # Quick simulation
└── README.md
```

## Running the Experiment

### Simulation (Recommended First)

```bash
# From project root
bash experiments/exp2/scripts/run_exp2_both.sh
```

Or manually:

```bash
# Shock A
python3 experiments/exp2/sim/exp2_sim.py \
  --n 1000 \
  --shock A_unavailable \
  --shock-point 500 \
  --seed 42

# Shock B
python3 experiments/exp2/sim/exp2_sim.py \
  --n 1000 \
  --shock A_degraded \
  --shock-point 500 \
  --seed 42
```

### Real Execution

Requires task data from symphony-data-generator:

```bash
python3 experiments/exp2/real/exp2_real.py \
  --tasks path/to/task_pool.jsonl \
  --n 200 \
  --shock A_unavailable \
  --shock-point 100
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n` | Number of tasks | 1000 |
| `--shock` | Shock type: `A_unavailable` or `A_degraded` | Required |
| `--shock-point` | Task index when shock occurs | n/2 |
| `--seed` | Random seed | 42 |
| `--topL` | Top-L selection | 3 |

## Expected Results

- LinUCB should detect the shock and adapt routing within ~50-100 tasks
- Recovery time depends on shock severity
- Static policies show no adaptation

## Output

Results saved to `sim/results/` or `real/results/`:
- `trajectory_*.csv`: Per-task selection history
- `summary.json`: Aggregate metrics
