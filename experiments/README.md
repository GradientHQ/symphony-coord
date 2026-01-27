# Symphony Experiments

This directory contains all experiments for the Symphony paper.

## Overview

| Experiment | Description | Type |
|------------|-------------|------|
| **Exp1** | Efficiency & Cost Analysis | Simulation + Real |
| **Exp2** | Robustness & Recovery under Role Shock | Simulation + Real |
| **Exp3** | System Performance Optimization | Simulation |

## Directory Structure

```
experiments/
├── README.md                 # This file
├── pretrain.py               # Main experiment runner
├── configs/                  # Configuration files
│   ├── openrouter/           # OpenRouter model configs
│   └── *.yaml                # Basic configs
├── scripts/                  # Shell scripts
│   ├── run_all_datasets.sh
│   ├── run_gsm8k_pretrain.sh
│   ├── run_bbh_pretrain.sh
│   ├── run_balanced_pretrain.sh
│   └── run_select_only.sh
├── exp1/                     # Efficiency & Cost
│   ├── sim/
│   └── real/
├── exp2/                     # Robustness & Recovery
│   ├── sim/
│   ├── real/
│   ├── plot/
│   └── scripts/
├── exp3/                     # System Optimization
│   └── ...
└── common/                   # Shared utilities
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Running Experiments

```bash
# Exp1: Efficiency & Cost
python experiments/exp1/sim/sim_efficiency_cost.py --n 1000

# Exp2: Robustness
bash experiments/exp2/scripts/run_exp2_both.sh

# Exp3: System Optimization
bash experiments/exp3/run_exp3.sh

# Main pretrain experiments
bash experiments/scripts/run_gsm8k_pretrain.sh
```

## Experiment Details

### Experiment 1: Efficiency & Cost Analysis

**Goal**: Compare agent selection strategies on efficiency and cost.

**Strategies**:
- Always-A: Always use strongest agent
- Static Rule: Fixed rule-based selection
- Random: Uniform random from candidates
- LinUCB: Adaptive bandit-based selection

See [exp1/README.md](exp1/real/README.md) for details.

### Experiment 2: Robustness & Recovery

**Goal**: Evaluate adaptation under non-stationary conditions.

**Shock Types**:
- `A_unavailable`: Agent becomes unavailable
- `A_degraded`: Agent performance degrades

See [exp2/README.md](exp2/README.md) for details.

### Experiment 3: System Performance Optimization

**Goal**: Evaluate routing optimization based on latency and load.

**Scenarios**:
- `latency_heterogeneous`: Different agent latencies
- `load_burst`: Dynamic load spikes
- `combined`: Both latency and load variations

See [exp3/README.md](exp3/README.md) for details.

## Main Pretrain Runner

The `pretrain.py` script supports:
- Multiple benchmarks (GSM8K, BBH, Medical QA)
- Three-phase execution (cold start, pretrain, test)
- LinUCB agent selection with online updates
- Multi-CoT voting

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--task-pool` | Path to JSONL task file | Required |
| `--benchmark` | Benchmark name | None |
| `--n` | Total tasks | 100 |
| `--cold-n` | Cold start phase | 10 |
| `--pretrain-n` | Pretrain phase | 70 |
| `--test-n` | Test phase | 20 |
| `--topL` | Top-L candidates | 3 |
| `--plan-k` | Number of plans | 3 |
| `--cot-count` | CoT paths per task | 3 |
| `--agents` | Agent IDs (comma-separated) | Required |
| `--runtime-dir` | Config directory | configs |

### Example

```bash
python experiments/pretrain.py \
  --task-pool data/gsm8k_full.jsonl \
  --benchmark gsm8k \
  --n 600 \
  --cold-n 200 \
  --pretrain-n 300 \
  --test-n 100 \
  --agents "deepseek-v3,openai-gpt-5-nano" \
  --runtime-dir experiments/configs
```

## Configuration

Agent configs are in `configs/openrouter/<model-name>/`:

```yaml
debug: false
role: "agent"
node_id: "agent-openrouter-016"
base_model: "openrouter:deepseek/deepseek-chat"
capabilities: [math, reasoning, code]
max_tokens: 512
temperature: 0.2
```

## Related Documentation

- [docs/OPENROUTER_CONFIG_GUIDE.md](../docs/OPENROUTER_CONFIG_GUIDE.md)
- [docs/RUN_ALL_DATASETS.md](../docs/RUN_ALL_DATASETS.md)
