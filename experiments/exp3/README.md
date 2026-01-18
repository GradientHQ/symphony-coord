# Exp3: Role Shock Robustness & Recovery

This experiment evaluates robustness and recovery under non-stationary "role shock" conditions. It is an **EXPERIMENT**, not a method contribution.

## Directory Structure

```
experiments/exp3/
├── sim/
│   ├── exp3_sim.py        # Simulation version (synthetic tasks, no real LLM calls)
│   └── results/
│       └── sim_exp3_robustness/
├── real/
│   ├── exp3_real.py       # Real execution version (real LLM/API calls)
│   └── results/
│       └── real_exp3_robustness/
├── plot/
│   └── plot_sim_vs_real.py  # Plot comparison between sim and real
├── run_exp3_both.sh       # Launcher script for sim
└── README.md              # This file
```

## Running Exp3 Simulation

### Option 1: Using the Launcher Script (Recommended)

```bash
cd /Users/caohuixi/symphony2.0
bash experiments/exp3/run_exp3_both.sh
```

This script will:
- Run both shock types (A_unavailable and A_degraded)
- Use default parameters: n=1000, shock_point=500, seed=42
- Save results to `experiments/exp3/sim/results/sim_exp3_robustness/`

### Option 2: Run Manually

Run a single shock type:

```bash
cd /Users/caohuixi/symphony2.0

# Shock A (A_unavailable)
python3 experiments/exp3/sim/exp3_sim.py \
  --n 1000 \
  --shock A_unavailable \
  --shock-point 500 \
  --seed 42 \
  --outdir experiments/exp3/sim/results/sim_exp3_robustness

# Shock B (A_degraded)
python3 experiments/exp3/sim/exp3_sim.py \
  --n 1000 \
  --shock A_degraded \
  --shock-point 500 \
  --seed 42 \
  --outdir experiments/exp3/sim/results/sim_exp3_robustness
```

### Simulation Parameters

- `--n`: Number of tasks (default: 1000)
- `--p-hard`: Probability of hard tasks (default: 0.2)
- `--shock`: Shock type, required: `A_unavailable` or `A_degraded`
- `--shock-point`: Task index when shock occurs (default: n//2)
- `--seed`: Random seed (default: 123)
- `--topL`: Top-L selection (default: 3)
- `--outdir`: Output directory

## Running Exp3 Real Execution

### Using Data Generator (Recommended)

1. **Generate tasks** (in `symphony-data-generator` directory):
   ```bash
   cd symphony-data-generator
   conda activate symphony-data-gen  # or use your environment
   python src/quick_start.py
   ```
   
   This generates `data/exp3/task_pool.jsonl`

2. **Run Exp3 real directly** (from project root):
   ```bash
   cd /Users/caohuixi/symphony2.0
   
   # Shock A (A_unavailable)
   python3 experiments/exp3/real/exp3_real.py \
     --tasks symphony-data-generator/data/exp3/task_pool.jsonl \
     --n 200 \
     --shock A_unavailable \
     --shock-point 100 \
     --seed 42 \
     --outdir experiments/exp3/real/results/real_exp3_robustness
   
   # Shock B (A_degraded)
   python3 experiments/exp3/real/exp3_real.py \
     --tasks symphony-data-generator/data/exp3/task_pool.jsonl \
     --n 200 \
     --shock A_degraded \
     --shock-point 100 \
     --seed 42 \
     --outdir experiments/exp3/real/results/real_exp3_robustness
   ```

**Important**: `exp3_real.py` **REQUIRES** the symphony-data-generator format:
- Required format: `{"benchmark": "...", "difficulty_bin": "easy"|"hard", "raw_data": {...}}`
- Real experiments MUST use real benchmark data from symphony-data-generator
- The sim version uses synthetic tasks; the real version uses real benchmark tasks

### Real Execution Parameters

- `--tasks`: **Required** - Path to JSONL file from symphony-data-generator (e.g., `symphony-data-generator/data/exp3/task_pool.jsonl`)
- `--n`: Number of tasks to use (default: 200)
- `--shock`: Shock type, required: `A_unavailable` or `A_degraded`
- `--shock-point`: Task index when shock occurs (default: n//2)
- `--seed`: Random seed (default: 42)
- `--outdir`: Output directory

## Output Format

### Simulation Output

Results are saved in timestamped directories:

```
sim_exp3_robustness/
├── ShockA/YYYY-MM-DD_HH-MM-SS/
│   ├── trajectory_linucb.csv
│   └── summary_linucb.json
└── ShockB/YYYY-MM-DD_HH-MM-SS/
    ├── trajectory_linucb.csv
    └── summary_linucb.json
```

### Real Execution Output

```
real_exp3_robustness/
├── A_unavailable/YYYY-MM-DD_HH-MM-SS/
│   └── trajectory_real.json
└── A_degraded/YYYY-MM-DD_HH-MM-SS/
    └── trajectory_real.json
```

## Comparing Sim vs Real

After running both sim and real experiments, use the plotting script:

```bash
cd /Users/caohuixi/symphony2.0

python3 experiments/exp3/plot/plot_sim_vs_real.py \
  --sim experiments/exp3/sim/results/sim_exp3_robustness/ShockA/YYYY-MM-DD_HH-MM-SS/trajectory_linucb.csv \
  --real experiments/exp3/real/results/real_exp3_robustness/A_unavailable/YYYY-MM-DD_HH-MM-SS/trajectory_real.json \
  --shock-point 500 \
  --out sim_vs_real_comparison.png
```

## Important Notes

1. **Core Algorithm**: The `GlobalLinUCB` and `build_x` functions in `core/linucb_selector.py` are shared and MUST NOT be changed for Exp3.

2. **Experiment Isolation**: Exp3 logic is experiment-specific and should NOT be imported by other experiments.

3. **Shock Types**:
   - `A_unavailable`: Agent A becomes unavailable
   - `A_degraded`: Agent A's performance degrades

4. **Real Execution**: Currently uses placeholder/dummy responses. Replace the `RealAgent.execute()` method with actual LLM API calls (OpenAI, vLLM, etc.) as needed.

