# Symphony

**A Decentralized Multi-Agent Framework for Edge Devices with Beacon-Guided Task Routing and CoT Voting**

Symphony is a decentralized multi-agent framework that enables intelligent agents to collaborate across heterogeneous edge devices through beacon-guided task routing and Chain-of-Thought (CoT) voting mechanisms.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Configuration Guide](#configuration-guide)
- [Citation](#citation)

## Overview

Symphony employs a three-stage pipeline:

1. **Planning Phase**: Multiple planning agents decompose complex queries into executable sub-tasks
2. **Execution Phase**: Beacon-guided routing matches sub-tasks to specialized agents using LinUCB-based selection
3. **Voting Phase**: CoT voting aggregates multiple agent responses for robust final answers

### Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  Planning Phase                         │
│  - Task decomposition (k plans)         │
│  - LinUCB plan selection                │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Execution Phase                        │
│  - Beacon broadcast for each sub-task   │
│  - Top-L agent candidate selection      │
│  - LinUCB agent selection               │
│  - Parallel CoT execution               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Voting Phase                           │
│  - CoT voting across responses          │
│  - Final answer aggregation             │
└─────────────────────────────────────────┘
    │
    ▼
Final Result
```

## Key Features

- **Decentralized Architecture**: No central orchestrator required, fault-tolerant
- **Intelligent Task Routing**: Beacon-based capability matching with LinUCB learning
- **Advanced Reasoning**: Multi-path CoT with majority voting
- **Edge-Optimized**: Runs on consumer-grade GPUs (RTX 3060/4090, Jetson, M-series Mac)

## Directory Structure

```
symphony/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Package configuration
│
├── core/                        # Core algorithms
│   ├── capability.py            # Capability matching
│   ├── linucb_selector.py       # LinUCB bandit selector
│   ├── routing.py               # Task routing
│   └── voting.py                # CoT voting mechanisms
│
├── agents/                      # Agent implementations
│   ├── agent.py                 # Main Agent class
│   └── user.py                  # User client
│
├── protocol/                    # Protocol definitions
│   ├── task_contract.py         # Task data structures
│   └── beacon.py                # Beacon messages
│
├── infra/                       # Infrastructure
│   └── ISEP.py                  # Service exchange protocol
│
├── models/                      # Model loaders
│   └── base_loader.py           # LLM loading utilities
│
├── symphony.py                  # Core orchestrator
├── main.py                      # Simple entry point
├── agent_register.py            # Agent registration runner
├── user_register.py             # User registration runner
│
├── experiments/                 # All experiments
│   ├── README.md                # Experiments overview
│   ├── pretrain.py              # Main experiment runner
│   ├── configs/                 # Configuration files
│   ├── scripts/                 # Shell scripts
│   ├── exp1/                    # Exp1: Efficiency & Cost
│   ├── exp2/                    # Exp2: Robustness & Recovery
│   └── exp3/                    # Exp3: System Optimization
│
├── scripts/                     # Utility scripts
│   ├── plotting/                # Visualization
│   │   ├── paper_figures/       # Paper figure generation
│   │   └── routing/             # Routing analysis plots
│   └── analysis/                # Analysis utilities
│
├── docs/                        # Documentation
├── examples/                    # Example configurations
└── tests/                       # Test suite
```

## Installation

### System Requirements

| Requirement | Minimum               | Recommended                 |
| ----------- | --------------------- | --------------------------- |
| Python      | 3.9                   | 3.10 or 3.11                |
| RAM         | 8 GB                  | 16 GB                       |
| GPU         | Optional              | CUDA-compatible (RTX 3060+) |
| OS          | Linux, macOS, Windows | Linux (Ubuntu 20.04+)       |

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/anonymous/symphony.git
cd symphony

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install core dependencies
pip install -r requirements.txt

# 5. Install Symphony in development mode
pip install -e .

# 6. Verify installation
python -c "import symphony; print('Symphony installed successfully')"
```

### Dependencies Overview

The `requirements.txt` includes:

**Core Dependencies** (required):
- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.30.0` - Hugging Face model library
- `numpy>=1.24.0` - Numerical computing
- `pyyaml>=6.0` - Configuration file parsing
- `requests>=2.28.0` - HTTP client for API calls
- `pyzmq>=25.0.0` - Distributed messaging
- `aiohttp>=3.8.0` - Async HTTP client

**Optional Dependencies** (for GPU acceleration):
- `accelerate>=0.20.0` - Distributed training
- `bitsandbytes>=0.41.0` - 8-bit quantization
- `peft>=0.4.0` - Parameter-efficient fine-tuning

### API Key Setup (Required for Real Experiments)

Symphony uses [OpenRouter](https://openrouter.ai/) for LLM API access:

```bash
# Option 1: Export in terminal (temporary)
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# Option 2: Add to shell profile (persistent)
echo 'export OPENROUTER_API_KEY="sk-or-v1-your-key-here"' >> ~/.bashrc
source ~/.bashrc

# Option 3: Create .env file (recommended for development)
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key-here' > .env
```

**Verify API key is set:**
```bash
python -c "import os; print('API Key configured' if os.getenv('OPENROUTER_API_KEY') else 'API Key NOT set')"
```

See [docs/OPENROUTER_CONFIG_GUIDE.md](docs/OPENROUTER_CONFIG_GUIDE.md) for detailed API setup instructions.

## Quick Start

### Running a Simple Task

```python
from symphony import SymphonyOrchestrator
from agents.agent import Agent

# Initialize orchestrator
orchestrator = SymphonyOrchestrator(
    agents=["agent1", "agent2", "agent3"],
    topL=3,
    cot_count=3
)

# Execute a task
result = orchestrator.run_task(
    task_description="Solve: What is 25 * 37?",
    requirements=["math"]
)

print(f"Result: {result['final_answer']}")
```

### Using OpenRouter API

```bash
# Set API key
export OPENROUTER_API_KEY="sk-or-v1-..."

# Run with OpenRouter models
python experiments/pretrain.py \
  --task-pool path/to/tasks.jsonl \
  --agents "deepseek-v3" \
  --runtime-dir experiments/configs \
  --n 100
```

## Running Experiments

All experiments are in the `experiments/` directory. See [experiments/README.md](experiments/README.md) for detailed documentation.

### Overview of Experiments

| Experiment   | Description                | Type              | Estimated Time                |
| ------------ | -------------------------- | ----------------- | ----------------------------- |
| **Exp1**     | Efficiency & Cost Analysis | Simulation + Real | 30 min (sim) / 2-4 hrs (real) |
| **Exp2**     | Robustness & Recovery      | Simulation + Real | 1-2 hrs                       |
| **Exp3**     | System Optimization        | Simulation        | 30 min                        |
| **Pretrain** | Main benchmark evaluation  | Real              | 4-8 hrs per benchmark         |

---

### Experiment 1: Efficiency & Cost Analysis

**Goal**: Compare agent selection strategies (Always-A, Static Rule, Random, LinUCB).

**Simulation Mode** (no API key needed):
```bash
cd experiments/exp1/sim
python sim_efficiency_cost.py --n 1000 --seed 42

# Output: Results saved to exp1_sim_results/
```

**Real Mode** (requires OpenRouter API key):
```bash
cd experiments/exp1/real
python exp1_real_openrouter.py --n 100

# Output: Results saved to exp1_real_results/
```

**Expected Output Files**:
- `accuracy_by_strategy.csv` - Accuracy comparison
- `cost_by_strategy.csv` - API cost comparison
- `selection_trace.json` - Agent selection decisions

---

### Experiment 2: Robustness & Recovery

**Goal**: Evaluate adaptation when agents become unavailable or degraded.

**Run both simulation and real:**
```bash
bash experiments/exp2/scripts/run_exp2_both.sh

# Or run separately:
python experiments/exp2/sim/exp2_sim.py --shock-type A_unavailable
python experiments/exp2/real/exp2_real.py --shock-type A_degraded
```

**Shock Types**:
- `A_unavailable`: Agent suddenly becomes unavailable
- `A_degraded`: Agent performance drops significantly

**Expected Output Files**:
- `recovery_curve.csv` - Accuracy over time after shock
- `adaptation_metrics.json` - Recovery time and final accuracy

---

### Experiment 3: System Optimization

**Goal**: Evaluate routing optimization under latency and load variations.

```bash
bash experiments/exp3/run_exp3.sh

# Or run directly:
python experiments/exp3/sim_system_optimization.py --scenario latency_heterogeneous
```

**Scenarios** (defined in `experiments/exp3/configs/scenarios.yaml`):
- `latency_heterogeneous`: Agents with different response latencies
- `load_burst`: Dynamic load spikes
- `combined`: Both latency and load variations

**Expected Output Files**:
- `latency_comparison.csv` - Response time metrics
- `load_balance_metrics.csv` - Task distribution across agents

---

### Main Pretrain Experiments (Benchmark Evaluation)

**Goal**: Evaluate Symphony on standard benchmarks (GSM8K, BBH, Medical QA).

**Run individual benchmarks:**
```bash
# GSM8K (math reasoning)
bash experiments/scripts/run_gsm8k_pretrain.sh

# BBH (Big-Bench Hard)
bash experiments/scripts/run_bbh_pretrain.sh

# Balanced sampling across all tasks
bash experiments/scripts/run_balanced_pretrain.sh

# All datasets sequentially
bash experiments/scripts/run_all_datasets.sh
```

**Run with custom parameters:**
```bash
python experiments/pretrain.py \
  --task-pool data/gsm8k_full.jsonl \
  --benchmark gsm8k \
  --n 600 \
  --cold-n 200 \
  --pretrain-n 300 \
  --test-n 100 \
  --topL 3 \
  --plan-k 3 \
  --cot-count 3 \
  --agents "deepseek-v3,openai-gpt-5-nano,openai-gpt-4-1-nano" \
  --runtime-dir experiments/configs
```

**Expected Output** (saved to `pretrain_results/<timestamp>/`):
- `accuracy_summary.csv` - Per-phase accuracy
- `ucb_trace.md` - LinUCB arm selection trace
- `progress_state.json` - Checkpoint for resumption

## Reproducing Paper Results

This section provides step-by-step instructions to reproduce all results in the paper.

### Step 1: Environment Setup

```bash
# Create fresh environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Set API key
export OPENROUTER_API_KEY="sk-or-v1-your-key"

# Verify setup
python -c "import symphony; import os; print('Ready!' if os.getenv('OPENROUTER_API_KEY') else 'Missing API key')"
```

### Step 2: Run All Experiments

```bash
# Exp1: Efficiency & Cost Analysis (Table 2 in paper)
python experiments/exp1/real/exp1_real_openrouter.py --n 2000

# Exp2: Robustness & Recovery (Figure 4 in paper)
bash experiments/exp2/scripts/run_all_experiments.sh

# Exp3: System Optimization (Figure 5 in paper)
bash experiments/exp3/run_exp3.sh

# Main Benchmark Results (Table 1 in paper)
bash experiments/scripts/run_all_datasets.sh
```

### Step 3: Generate Paper Figures

```bash
# Figure 3: Robustness bar charts
python scripts/plotting/paper_figures/plot_robustness_bars.py

# Figure 4: 3D robustness surface
python scripts/plotting/paper_figures/plot_robustness_3d_surface.py

# Figure 5: Gap analysis
python scripts/plotting/paper_figures/plot_gap_analysis.py

# Figure 6: Parallel coordinates
python scripts/plotting/paper_figures/plot_parallel_coordinates.py

# Routing analysis visualizations
python scripts/plotting/routing/plot_from_json.py pretrain_results/<your-result-dir>
python scripts/plotting/routing/plot_agent_donut.py pretrain_results/<your-result-dir>
```

### Expected Results Summary

| Experiment        | Key Metric                         | Expected Range |
| ----------------- | ---------------------------------- | -------------- |
| Exp1 (Efficiency) | LinUCB vs Always-A cost reduction  | 15-25%         |
| Exp2 (Robustness) | Recovery time after shock          | < 50 tasks     |
| Exp3 (Latency)    | Load-balanced vs naive improvement | 10-20%         |
| GSM8K             | Test accuracy (LinUCB)             | 75-85%         |
| BBH               | Macro-average accuracy             | 60-70%         |

---

## Troubleshooting

### Common Issues

**1. `ModuleNotFoundError: No module named 'symphony'`**
```bash
# Ensure you're in the project root and installed in dev mode
pip install -e .
```

**2. `OPENROUTER_API_KEY not set`**
```bash
# Check if key is exported
echo $OPENROUTER_API_KEY

# If empty, set it
export OPENROUTER_API_KEY="sk-or-v1-your-key"
```

**3. `CUDA out of memory`**
```bash
# Use CPU-only mode or reduce batch size
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

**4. `Connection timeout` or `Rate limit exceeded`**
```bash
# Reduce concurrent requests in config
# Edit experiments/configs/openrouter/<model>/config_*.yaml
# Add: rate_limit_delay: 1.0
```

**5. `FileNotFoundError: task-pool not found`**
```bash
# Ensure task data files exist
# Download from paper supplementary materials or generate:
python scripts/analysis/balanced_task_pool.py --output data/tasks.jsonl
```

### Getting Help

- Check [experiments/README.md](experiments/README.md) for experiment-specific issues
- Check [docs/OPENROUTER_CONFIG_GUIDE.md](docs/OPENROUTER_CONFIG_GUIDE.md) for API setup
- Verify Python version: `python --version` (requires 3.9+)

## Configuration Guide

### Agent Configuration

Configs in `experiments/configs/openrouter/<model>/`:

```yaml
debug: false
role: "agent"
node_id: "agent-openrouter-016"
base_model: "openrouter:deepseek/deepseek-chat"
capabilities: [math, reasoning, code]
max_tokens: 512
temperature: 0.2
```

### Key Experiment Parameters

| Parameter     | Description       | Default  |
| ------------- | ----------------- | -------- |
| `--task-pool` | Task JSONL file   | Required |
| `--n`         | Total tasks       | 100      |
| `--topL`      | Top-L candidates  | 3        |
| `--plan-k`    | Plans to generate | 3        |
| `--cot-count` | CoT paths         | 3        |
| `--agents`    | Agent IDs         | Required |

See [docs/OPENROUTER_CONFIG_GUIDE.md](docs/OPENROUTER_CONFIG_GUIDE.md) for detailed setup.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
