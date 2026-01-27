# Exp1 Real OpenRouter: Efficiency & Cost

This is the real version of Exp1, using real OpenRouter agents (loaded from config files) instead of simulated agents. All other logic (multi-strategy comparison, reward shaping, fallback, drift, etc.) remains consistent with the simulation version.

## Environment Setup (Must Complete First)

Before running experiments, **you must set up the Python environment**:

### 1. Create and Activate Virtual Environment

```bash
# Navigate to project root
cd /path/to/symphony

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install main project dependencies
pip install -r requirements.txt

# Install symphony-data-generator dependencies (for loading HumanEval/GSM8K)
pip install -r symphony-data-generator/requirements.txt
```

### 3. Set OpenRouter API Key

```bash
# Temporary setup (current terminal session)
export OPENROUTER_API_KEY="sk-or-v1-..."
# Or permanent setup (add to ~/.zshrc or ~/.bashrc)
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.zshrc
source ~/.zshrc
```

### 4. Verify Environment

```bash
# Check key dependencies
python3 -c "import yaml, numpy, pandas, json; print('✅ Core libraries OK')"
python3 -c "from datasets import load_dataset; print('✅ datasets OK')"
python3 -c "import requests; print('✅ requests OK')"

# Check API Key
python3 -c "import os; key = os.getenv('OPENROUTER_API_KEY'); print('✅ API Key set' if key else '❌ API Key not set')"
```

### 5. Prepare Datasets (if needed)

```bash
# If datasets haven't been generated yet, run:
cd symphony-data-generator
python src/quick_start.py
cd ..
```

## Prerequisites

1. **Python Environment**: Virtual environment set up with all dependencies installed (see above)
2. **API Key**: `OPENROUTER_API_KEY` environment variable set
3. **Config Files**: Ensure the following config files exist in `runtime/configs/openrouter/<agent-name>/` directory:
   - `config_agent_openrouter_1.yaml` (Agent A)
   - `config_agent_openrouter_2.yaml` (Agent B)
   - `config_agent_openrouter_3.yaml` (Agent C)
   - `config_agent_openrouter_4.yaml` (Agent D)
   - `config_agent_openrouter_5.yaml` (Agent E)
   - `config_agent_openrouter_6.yaml` (Agent F)
   - `config_agent_openrouter_7.yaml` (Agent G)

## Quick Start

### Basic Run (Using Defaults)

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # if using venv

# Run with all default values (recommended)
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py

# Equivalent to:
# --n 2000 --seed 1 --topL 3 --agents 1,2,3,4,5,6,7 --benchmarks humaneval,gsm8k
```

### Small-Scale Test

```bash
# Run 20-task small-scale test
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --n 20 \
  --agents 1,2,3
```

### Full Experiment (Using Config File Settings)

```bash
# Run full 2000-task experiment (using defaults)
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py

# Or specify parameters
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --n 2000 \
  --seed 1 \
  --topL 3 \
  --agents 1,2,3,4,5,6,7 \
  --benchmarks humaneval,gsm8k
```

### Single Benchmark Only

```bash
# HumanEval only
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --benchmarks humaneval

# GSM8K only
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --benchmarks gsm8k
```

### Experiment with Drift

```bash
# Run experiment with drift enabled
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --n 1000 \
  --p-hard 0.2 \
  --seed 123 \
  --topL 2 \
  --fallback \
  --drift \
  --agents 1,2,3,4,5,6
```

## Parameter Reference

### Basic Parameters
- `--n`: Number of tasks (default: `2000`, from config_exp1.yaml)
- `--p-hard`: Hard task probability (default: `0.2`, 80/20 split)
- `--seed`: Random seed (default: `1`)
- `--topL`: Top-L candidate selection (default: `3`, from config_exp1.yaml)
- `--benchmarks`: Benchmark list, comma-separated (default: `humaneval,gsm8k`)
  - Options: `humaneval`, `gsm8k`, or `humaneval,gsm8k`
- `--outdir`: Output directory (default: `experiments/exp1_real_openrouter/results`)
- `--no-plots`: Don't generate plots
- `--config-dir`: Config file directory (default: `runtime/configs/openrouter`)
- `--agents`: Agent ID list, comma-separated (default: `1,2,3,4,5,6,7`, all 7 agents)

### LinUCB Parameters
- `--alpha`: LinUCB exploration scale (default: `1.0`, from config_exp1.yaml)
- `--l2`: LinUCB L2 regularization (default: `1.0`, from config_exp1.yaml)
- `--delta`: LinUCB confidence (default: `0.1`, from config_exp1.yaml)
- `--S`: Upper bound of ||theta*|| (default: `1.0`, from config_exp1.yaml)

### Reward Shaping Parameters
- `--latency-scale-ms`: Latency normalization scale (default: `2000.0`)
- `--latency-penalty`: Latency penalty multiplier (default: `0.05`, from config_exp1.yaml)
- `--cost-lambda`: Cost penalty multiplier (default: `0.3`, from config_exp1.yaml)

### Realism Switches
- `--fallback`: If selected agent fails, fallback to strong agent A
- `--drift`: Enable non-stationary drift (t>=500, currently ignored for real agents)

## Output

Results are saved in `experiments/exp1_real_openrouter/results/YYYY-MM-DD_HH-MM-SS/` directory:

- `summary.csv`: Summary statistics for each strategy (cost / success / latency / fallback, etc.)
- `trajectory_always_A.csv`: Always-A strategy step-by-step trajectory
- `trajectory_static_rule.csv`: Static rule strategy step-by-step trajectory
- `trajectory_random.csv`: Random strategy step-by-step trajectory
- `trajectory_linucb.csv`: LinUCB strategy step-by-step trajectory
- `plot_cost_success_pretty.png`: Cost and success rate bar chart
- `plot_all_cum_cost_clean.png`: Cumulative average cost curves
- `plot_all_cum_success_clean.png`: Cumulative average success rate curves

## Example Output

During runtime you'll see output similar to:

```
============================================================
Loading OpenRouter Agents
============================================================
✅ Loaded agent A: agent-openrouter-001 (openrouter:google/gemini-2.5-flash-lite)
✅ Loaded agent B: agent-openrouter-002 (openrouter:openai/gpt-5-nano)
✅ Loaded agent C: agent-openrouter-003 (openrouter:openai/gpt-4o-mini)
...

✅ Loaded 6 agents

============================================================
Generating Tasks
============================================================
✅ Generated 1000 tasks

============================================================
Running Policies
============================================================

[always_A] Starting...
[always_A] Task 10/1000, Success: 9/10 (90.0%)
[always_A] Task 20/1000, Success: 18/20 (90.0%)
...
[always_A] success=0.950 total_cost=150.0 avg_cost=0.150 avg_lat=900.0ms fallback=0.000 choose(A,B,C,D,E)=(1000,0,0,0,0)

[static_rule] Starting...
...
[linucb] Starting...
...
```

## Comparison with Simulation Version

- **Similarities**:
  - All strategy logic is identical
  - Reward shaping parameters are consistent
  - LinUCB parameters are consistent
  - State update logic is consistent

- **Differences**:
  - Simulation version uses `SimAgent` (simulated execution)
  - Real version uses `RealAgentWrapper` (calls real OpenRouter API)
  - Real version requires API key and network connection
  - Real version execution time depends on API response time

## Notes

1. **API Cost**: Real version calls OpenRouter API, incurring actual costs. Recommend small-scale testing (`--n 20`) first to verify setup.

2. **Network Connection**: Ensure network can access `openrouter.ai`.

3. **API Key**: Ensure `OPENROUTER_API_KEY` environment variable is set.

4. **Execution Time**: Real version execution time depends on API response time; 1000 tasks may take significant time.

5. **Error Handling**: If an agent call fails, error message is printed and execution continues (failed tasks are marked as failures).

## Troubleshooting

If you encounter issues:

1. **Check API Key**:
   ```bash
   echo $OPENROUTER_API_KEY
   ```

2. **Test Configuration**:
   ```bash
   python3 test_openrouter_setup.py
   ```

3. **View Error Messages**: Runtime will print detailed error messages, including failed API calls.
