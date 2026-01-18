# Symphony 2.0 Data Generator

A streamlined data generation pipeline for creating experiment-ready task pools with accurate difficulty scoring and full-dataset threshold computation.

## 🎯 Overview

The Symphony Data Generator provides a unified framework for:
- **Loading and preprocessing** 5 major benchmarks (HumanEval, GSM8K, BBH, AMC, Medical QA)
- **Computing difficulty scores** using data-driven normalization constants
- **Generating experiment streams** with custom difficulty distributions
- **Using full-dataset thresholds** for stable, accurate difficulty binning

## 🏗️ Architecture

### Two-Stage Workflow

```
Stage 1: Preprocessing (One-time)
  └─> Downloads full datasets
  └─> Computes difficulty scores
  └─> Calculates normalization constants (95th percentile)
  └─> Saves to data/benchmarks/full/

Stage 2: Experiment Generation (Fast, repeatable)
  └─> Loads full preprocessed datasets
  └─> Computes thresholds from full datasets
  └─> Samples tasks directly for experiments
  └─> Saves to data/exp{1,2,3,5}/
```

### Key Features

- ✅ **Full-dataset thresholds**: Stable, accurate difficulty binning
- ✅ **Data-driven normalization**: Uses 95th percentile of feature distributions
- ✅ **Within-benchmark normalization**: Scores comparable within each domain
- ✅ **Direct experiment generation**: No intermediate sampling steps
- ✅ **Per-benchmark sampling**: Maintains benchmark identity

## 📁 Project Structure

```
symphony-data-generator/
├── config/
│   └── data_config.yaml          # Main configuration file
├── src/
│   ├── data_generator.py         # Core module (DatasetBuilder, Scorers)
│   └── quick_start.py            # Quick start script
├── doc/
│   └── README.md                 # This file
├── data/                         # Generated data (created at runtime)
│   ├── benchmarks/
│   │   └── full/                 # Full preprocessed datasets
│   └── exp{1,2,3,5}/            # Experiment task pools
├── environment.yaml              # Conda environment specification
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yaml

# Activate environment
conda activate symphony-data-gen

# Run data generator
python src/quick_start.py
```

**Note**: The conda environment is named `symphony-data-gen` (from `environment.yaml`)

#### Option 2: Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Run data generator
python src/quick_start.py
```

### Basic Usage

```python
from src.data_generator import DatasetBuilder

# Initialize
builder = DatasetBuilder('config/data_config.yaml')

# Step 1: Preprocess all benchmarks (one-time)
builder.preprocess_all_benchmarks(
    output_dir='data/benchmarks/full',
    force_reprocess=False,
)

# Step 2: Generate experiment stream with explicit task allocation
tasks = builder.build_task_stream(
    benchmarks_to_include=['humaneval', 'gsm8k'],
    benchmark_ratios={'humaneval': 0.5, 'gsm8k': 0.5},  # 50% from each
    difficulty_split='50:50',  # 50% easy, 50% hard
    n_total_tasks=1000,
    random_seed=2025,
)

# Save results
builder.save_task_pool(tasks, 'data/my_experiment/task_pool.jsonl')
builder.save_statistics(tasks, 'data/my_experiment/statistics.json')
```

## 📊 Supported Benchmarks

| Benchmark      | Type                | Full Dataset Size | Difficulty Metric                   |
| -------------- | ------------------- | ----------------- | ----------------------------------- |
| **HumanEval**  | Code Generation     | 164 tasks         | Test complexity + prompt length     |
| **GSM8K**      | Math Word Problems  | 1,319 tasks       | Reasoning steps                     |
| **BBH**        | Multi-hop Reasoning | 2,437 tasks       | Task complexity + input length      |
| **AMC**        | Math Competition    | 83 tasks          | Problem number (1-30)               |
| **Medical QA** | Domain QA           | 1,273 tasks       | Reasoning steps + keywords + length |

## ⚙️ Configuration

### `config/data_config.yaml`

This is the main configuration file. You typically only need to modify this to enable/disable benchmarks.

```yaml
# Enable/disable benchmarks
benchmarks:
  humaneval:
    enabled: true           # Set to false to disable this benchmark
    path: datasets:openai_humaneval
  gsm8k:
    enabled: true
    path: datasets:gsm8k/main
  bbh:
    enabled: true
    path: datasets:lukaemon/bbh
  amc:
    enabled: true
    path: datasets:AI-MO/aimo-validation-amc
  medical_qa:
    enabled: true
    path: datasets:GBaker/MedQA-USMLE-4-options

# Stream generation settings
stream_generation:
  # Difficulty percentile thresholds: [low_pct, high_pct]
  # Tasks with score ≤ low_pct are "easy", score ≥ high_pct are "hard"
  # These thresholds are computed from the FULL dataset for stability
  difficulty_percentiles:
    humaneval: [20, 80]    # Bottom 20% = easy, top 20% = hard
    gsm8k: [20, 80]        # Adjust these numbers to change thresholds
    bbh: [20, 80]          # e.g., [10, 90] = stricter (only extremes)
    amc: [20, 80]          # e.g., [30, 70] = looser (more in easy/hard)
    medical_qa: [20, 80]

  # Whether to allow duplicate tasks in experiments
  # Set to false to avoid duplicates (recommended for large datasets)
  # Set to true if you need more tasks than available in easy/hard bins
  sample_with_replacement: false

# Mixing rules
mixing_rules:
  # Prevent certain benchmark combinations (if needed)
  exclude_combinations: []
  # Example: Uncomment to prevent mixing code and math competition
  # - [humaneval, amc]
```

### Configuration Parameters Explained

#### `benchmarks.*.enabled`
- **What**: Enables/disables a benchmark for preprocessing and experiments
- **Values**: `true` or `false`
- **When to change**: Disable benchmarks you don't need to save time/space

#### `difficulty_percentiles`
- **What**: Defines what percentile ranges are "easy" and "hard"
- **Values**: `[low_percentile, high_percentile]` (0-100)
- **Default**: `[20, 80]` (bottom 20% easy, top 20% hard)
- **Examples**:
  - `[10, 90]`: Stricter (only extreme tasks are easy/hard)
  - `[30, 70]`: Looser (more tasks classified as easy/hard)
  - `[50, 50]`: Split at median (NOT recommended - use default)

#### `sample_with_replacement`
- **What**: Whether to allow duplicate tasks when sampling
- **Values**: `true` or `false`
- **Default**: `false`
- **When to use true**: When you need more tasks than available (e.g., HumanEval has only 33 easy tasks, but you need 100)

#### `exclude_combinations`
- **What**: List of benchmark pairs that shouldn't be mixed
- **Values**: List of `[benchmark1, benchmark2]` pairs
- **Example**: `- [humaneval, amc]` prevents mixing code and math competition

## 🔬 How It Works

### 1. Preprocessing Stage

**Purpose**: Download, score, and normalize all benchmark tasks.

**Process**:
1. Downloads full datasets from HuggingFace
2. Computes raw difficulty scores using benchmark-specific algorithms
3. Calculates **data-driven normalization constants** (95th percentile of feature distributions)
4. Normalizes scores within each benchmark (0.0-1.0 range)
5. Saves to `data/benchmarks/full/{benchmark}_full.jsonl`

**Example**:
```python
# HumanEval normalization constants (from full dataset)
norm_constants = {
    'max_asserts': 15,      # 95th percentile of assert counts
    'max_prompt_len': 450,   # 95th percentile of prompt lengths
}

# Raw score calculation
raw_score = 0.6 * (n_asserts / norm_constants['max_asserts']) + \
            0.4 * (prompt_len / norm_constants['max_prompt_len'])

# Normalized score (0.0-1.0)
normalized_score = min(raw_score, 1.0)
```

### 2. Experiment Generation Stage

**Purpose**: Generate experiment-ready task pools with accurate difficulty bins.

**Process**:
1. **Load full datasets** from `data/benchmarks/full/`
2. **Compute thresholds** from FULL datasets (stable, accurate)
   ```python
   # Example: HumanEval thresholds
   scores = [t.difficulty_score for t in all_humaneval_tasks]  # All 164 tasks
   threshold_easy = np.percentile(scores, 20)   # 0.194
   threshold_hard = np.percentile(scores, 80)  # 0.633
   ```
3. **Assign bins** using full thresholds
   - Easy: score ≤ threshold_easy
   - Hard: score ≥ threshold_hard
   - Medium: threshold_easy < score < threshold_hard (not used)
4. **Sample tasks** directly from full datasets
   - Calculate per-benchmark allocation (e.g., 400 tasks each)
   - Sample easy/hard tasks according to `difficulty_split`
   - Use sampling with/without replacement as needed
5. **Clone and shuffle** tasks to create final stream
6. **Save** to `data/exp{X}/`

### Why Full-Dataset Thresholds?

**Problem with sampled thresholds**:
- Unstable: Thresholds vary with different samples
- Unrepresentative: Small samples miss edge cases
- Hard to reproduce: Different experiments get different thresholds

**Solution: Full-dataset thresholds**:
- ✅ Stable: Same thresholds across all experiments
- ✅ Representative: Captures full difficulty spectrum
- ✅ Reproducible: Consistent binning every time
- ✅ Accurate: Based on complete data distribution

## 📝 Task Data Structure

### Task Object

```python
@dataclass
class Task:
    task_id: str              # Unique identifier
    benchmark: str            # Source benchmark (humaneval, gsm8k, etc.)
    difficulty_score: float   # Normalized 0.0-1.0 (within benchmark)
    difficulty_bin: str       # easy/hard (based on full-dataset thresholds)
    raw_data: Dict           # Original task data from HuggingFace
    scorer_metadata: Dict     # Features used in difficulty scoring
```

### JSONL Format (task_pool.jsonl)

```json
{
    "task_id": "exp_00000_humaneval",
    "benchmark": "humaneval",
    "difficulty_score": 0.350,
    "difficulty_bin": "easy",
    "raw_data": {
        "task_id": "HumanEval/0",
        "prompt": "from typing import List\ndef has_close_elements(...)",
        "entry_point": "has_close_elements",
        "test": "assert has_close_elements([1.0, 2.0], 0.5) == False"
    },
    "scorer_metadata": {
        "n_asserts": 5,
        "prompt_len": 42
    }
}
```

### Statistics Format (statistics.json)

```json
{
    "n_total": 2000,
    "benchmarks": ["humaneval", "gsm8k", "bbh", "amc", "medical_qa"],
    "difficulty_distribution": {
        "mean": 0.456,
        "std": 0.369,
        "min": 0.0,
        "max": 1.0
    },
    "difficulty_bins": {
        "easy": 1000,
        "hard": 1000
    },
    "benchmark_breakdown": {
        "humaneval": {
            "count": 400,
            "mean_difficulty": 0.439,
            "easy": 200,
            "hard": 200
        },
        "gsm8k": {
            "count": 400,
            "mean_difficulty": 0.461,
            "easy": 200,
            "hard": 200
        }
    }
}
```

## 🔧 API Reference

### DatasetBuilder

Main class for data generation and preprocessing.

#### Initialization

```python
builder = DatasetBuilder(config_path='config/data_config.yaml')
```

#### Methods

##### `preprocess_all_benchmarks()`

Preprocess all enabled benchmarks (one-time operation).

```python
results = builder.preprocess_all_benchmarks(
    output_dir='data/benchmarks/full',
    force_reprocess=False,  # Set True to reprocess even if cached
)
```

**Returns**: `Dict[str, List[Task]]` mapping benchmark names to preprocessed tasks

**Output**:
- `{benchmark}_full.jsonl`: All tasks with difficulty scores
- `{benchmark}_full_meta.json`: Metadata, normalization constants, config fingerprint

##### `build_task_stream()`

Generate experiment task stream directly from full datasets.

```python
tasks = builder.build_task_stream(
    benchmarks_to_include=['humaneval', 'gsm8k'],
    difficulty_split='50:50',  # easy_pct:hard_pct
    n_total_tasks=1000,
    random_seed=2025,
    difficulty_percentiles=None,  # Optional: override config defaults
    benchmark_ratios=None,         # Optional: override config defaults
    sample_with_replacement=None,  # Optional: override config default
)
```

**Parameters**:
- `benchmarks_to_include`: List of benchmark names to include
- `difficulty_split`: String like "50:50" or "80:20" (easy:hard ratio)
- `n_total_tasks`: Total number of tasks in experiment stream
- `random_seed`: Random seed for reproducibility
- `benchmark_ratios`: Dict `{benchmark: ratio}` summing to 1.0 (if None, uses equal split)
- `difficulty_percentiles`: Optional dict `{benchmark: [low_pct, high_pct]}`
- `sample_with_replacement`: Optional bool (default: from config)

**Returns**: `List[Task]` ready for experiments

##### `save_task_pool()`

Save task stream to JSONL file.

```python
builder.save_task_pool(
    task_stream=tasks,
    output_path='data/exp1/task_pool.jsonl'
)
```

##### `save_statistics()`

Save difficulty statistics to JSON file.

```python
builder.save_statistics(
    task_stream=tasks,
    output_path='data/exp1/statistics.json'
)
```

## 💡 Usage Examples

### Example 1: Quick Start (All Experiments)

Run the default experiments defined in `quick_start.py`:

```bash
# Activate conda environment
conda activate symphony-data-gen

# Run quick start script
python src/quick_start.py
```

This generates:
- `data/exp1/`: 1000 tasks (80:20 easy:hard, humaneval + gsm8k, 500 each)
- `data/exp2/`: 500 tasks (50:50 easy:hard, humaneval + gsm8k, 250 each)
- `data/exp3/`: 500 tasks (50:50 easy:hard, humaneval + gsm8k, 250 each)
- `data/exp5/`: 2000 tasks (50:50 easy:hard, all 5 benchmarks, 400 each)

### Example 2: Custom Experiment

```python
from src.data_generator import DatasetBuilder

builder = DatasetBuilder('config/data_config.yaml')

# Preprocess (if not already done)
builder.preprocess_all_benchmarks(output_dir='data/benchmarks/full')

# Generate custom experiment with explicit task allocation
tasks = builder.build_task_stream(
    benchmarks_to_include=['humaneval', 'gsm8k', 'bbh'],
    benchmark_ratios={'humaneval': 0.4, 'gsm8k': 0.4, 'bbh': 0.2},  # Must sum to 1.0
    difficulty_split='70:30',  # 70% easy, 30% hard
    n_total_tasks=1500,
    random_seed=42,
)

# Save
builder.save_task_pool(tasks, 'data/custom_exp/task_pool.jsonl')
builder.save_statistics(tasks, 'data/custom_exp/statistics.json')
```

### Example 3: Reading Task Pools

```python
import json

# Read task pool
tasks = []
with open('data/exp1/task_pool.jsonl') as f:
    for line in f:
        task = json.loads(line)
        tasks.append(task)

# Filter by difficulty
easy_tasks = [t for t in tasks if t['difficulty_bin'] == 'easy']
hard_tasks = [t for t in tasks if t['difficulty_bin'] == 'hard']

# Route by benchmark
for task in tasks:
    benchmark = task['benchmark']
    raw_data = task['raw_data']
    
    if benchmark == 'humaneval':
        # Use your HumanEval evaluator
        result = evaluate_humaneval(raw_data)
    elif benchmark == 'gsm8k':
        # Use your GSM8K evaluator
        result = evaluate_gsm8k(raw_data)
    elif benchmark == 'bbh':
        # Use your BBH evaluator
        result = evaluate_bbh(raw_data)
    
    # Log result with task metadata
    log_result(
        task_id=task['task_id'],
        benchmark=benchmark,
        difficulty_bin=task['difficulty_bin'],
        difficulty_score=task['difficulty_score'],
        result=result
    )
```

### Example 4: Custom Difficulty Thresholds

```python
# Use different percentiles for specific experiment
tasks = builder.build_task_stream(
    benchmarks_to_include=['humaneval'],
    benchmark_ratios={'humaneval': 1.0},  # All tasks from humaneval
    difficulty_split='50:50',
    n_total_tasks=100,
    difficulty_percentiles={
        'humaneval': [10, 90],  # Bottom 10% = easy, top 10% = hard
    }
)
```

### Example 5: Automatic Equal Split

```python
# If benchmark_ratios not provided, defaults to equal split
tasks = builder.build_task_stream(
    benchmarks_to_include=['humaneval', 'gsm8k', 'bbh'],
    # No benchmark_ratios specified -> automatic equal split (33.3% each)
    difficulty_split='50:50',
    n_total_tasks=300,
    random_seed=42,
)
```

## 🎮 Customizing Experiments

### How to Modify Existing Experiments

The easiest way to customize experiments is to edit `src/quick_start.py`. Here's how:

#### Scenario 1: Change Number of Tasks

```python
# In src/quick_start.py, find the experiments dict:
experiments = {
    'exp1': {
        'name': 'Exp 1: Routing Efficiency (80% easy, 20% hard)',
        'benchmarks': ['humaneval', 'gsm8k'],
        'benchmark_ratios': {'humaneval': 0.5, 'gsm8k': 0.5},
        'difficulty_split': '80:20',
        'n_total_tasks': 2000,  # ← Change from 1000 to 2000
        'random_seed': 2025,
    },
}
```

#### Scenario 2: Change Difficulty Split

```python
'exp1': {
    # ...
    'difficulty_split': '90:10',  # ← Change to 90% easy, 10% hard
    # ...
}
```

#### Scenario 3: Change Benchmark Allocation

```python
'exp1': {
    'benchmarks': ['humaneval', 'gsm8k'],
    'benchmark_ratios': {'humaneval': 0.7, 'gsm8k': 0.3},  # ← 70% humaneval, 30% gsm8k
    # Must sum to 1.0!
}
```

#### Scenario 4: Add a New Experiment

```python
experiments = {
    # ... existing experiments ...
    'exp6': {  # ← Add new experiment
        'name': 'Exp 6: My Custom Experiment',
        'benchmarks': ['bbh', 'medical_qa'],
        'benchmark_ratios': {'bbh': 0.6, 'medical_qa': 0.4},
        'difficulty_split': '60:40',  # 60% easy, 40% hard
        'n_total_tasks': 500,
        'random_seed': 2025,
    },
}
```

#### Scenario 5: Use Only One Benchmark

```python
'exp7': {
    'name': 'Exp 7: HumanEval Only',
    'benchmarks': ['humaneval'],
    'benchmark_ratios': {'humaneval': 1.0},  # ← All tasks from humaneval
    'difficulty_split': '50:50',
    'n_total_tasks': 150,  # HumanEval has 164 total, 33 easy, 33 hard
    'random_seed': 2025,
}
```

### Common Research Scenarios

#### Scenario A: Test Easy vs Hard Performance

```python
# Create two experiments: one with only easy tasks, one with only hard
experiments = {
    'easy_only': {
        'name': 'Easy Tasks Only',
        'benchmarks': ['humaneval', 'gsm8k'],
        'benchmark_ratios': {'humaneval': 0.5, 'gsm8k': 0.5},
        'difficulty_split': '100:0',  # ← 100% easy, 0% hard
        'n_total_tasks': 200,
        'random_seed': 2025,
    },
    'hard_only': {
        'name': 'Hard Tasks Only',
        'benchmarks': ['humaneval', 'gsm8k'],
        'benchmark_ratios': {'humaneval': 0.5, 'gsm8k': 0.5},
        'difficulty_split': '0:100',  # ← 0% easy, 100% hard
        'n_total_tasks': 200,
        'random_seed': 2025,
    },
}
```

#### Scenario B: Progressive Difficulty

```python
# Create a series of experiments with increasing difficulty
experiments = {
    'prog_1': {
        'name': 'Progressive Difficulty - Level 1',
        'benchmarks': ['gsm8k'],
        'benchmark_ratios': {'gsm8k': 1.0},
        'difficulty_split': '90:10',  # Mostly easy
        'n_total_tasks': 100,
        'random_seed': 2025,
    },
    'prog_2': {
        'name': 'Progressive Difficulty - Level 2',
        'benchmarks': ['gsm8k'],
        'benchmark_ratios': {'gsm8k': 1.0},
        'difficulty_split': '70:30',  # Balanced toward easy
        'n_total_tasks': 100,
        'random_seed': 2025,
    },
    'prog_3': {
        'name': 'Progressive Difficulty - Level 3',
        'benchmarks': ['gsm8k'],
        'benchmark_ratios': {'gsm8k': 1.0},
        'difficulty_split': '30:70',  # Balanced toward hard
        'n_total_tasks': 100,
        'random_seed': 2025,
    },
}
```

#### Scenario C: Domain-Specific Testing

```python
# Test performance on different task types
experiments = {
    'coding': {
        'name': 'Coding Tasks',
        'benchmarks': ['humaneval'],
        'benchmark_ratios': {'humaneval': 1.0},
        'difficulty_split': '50:50',
        'n_total_tasks': 150,
        'random_seed': 2025,
    },
    'math': {
        'name': 'Math Tasks',
        'benchmarks': ['gsm8k', 'amc'],
        'benchmark_ratios': {'gsm8k': 0.7, 'amc': 0.3},
        'difficulty_split': '50:50',
        'n_total_tasks': 300,
        'random_seed': 2025,
    },
    'reasoning': {
        'name': 'Reasoning Tasks',
        'benchmarks': ['bbh'],
        'benchmark_ratios': {'bbh': 1.0},
        'difficulty_split': '50:50',
        'n_total_tasks': 200,
        'random_seed': 2025,
    },
}
```

### Important Notes for Researchers

**⚠️ Task Availability Constraints**:
- **HumanEval**: Only 164 total tasks (33 easy, 33 hard with default [20,80] thresholds)
  - If you need more, set `sample_with_replacement: true` in config
  - Or use different percentiles (e.g., [30,70] gives more easy/hard tasks)
- **GSM8K**: 1,319 total tasks (plenty of easy/hard tasks available)
- **BBH**: 2,437 total tasks (plenty available)
- **AMC**: Only 83 total tasks (18 easy, 18 hard - very limited!)
- **Medical QA**: 1,273 total tasks (plenty available)

**🔧 Adjusting Difficulty Thresholds**:

If you need more easy/hard tasks, modify `config/data_config.yaml`:

```yaml
difficulty_percentiles:
  humaneval: [30, 70]  # Changed from [20, 80] - gives MORE easy/hard tasks
  # [30, 70] means: bottom 30% = easy, top 30% = hard
  # This gives you ~49 easy and ~49 hard tasks instead of 33 each
```

**🎲 Random Seeds**:
- Same seed = same tasks selected (reproducible)
- Different seed = different random sample
- Useful for creating multiple variations of the same experiment

## 🐛 Troubleshooting

### Issue: FileNotFoundError for preprocessed data

**Solution**: Run preprocessing first:
```python
builder.preprocess_all_benchmarks(output_dir='data/benchmarks/full')
```

### Issue: SSL/TLS errors when downloading datasets

**Solution**: The code uses `download_mode='force_redownload'` and a local cache directory. If issues persist:
1. Check network connection
2. Try running with `force_reprocess=True` to clear cache
3. Check HuggingFace dataset availability

### Issue: Sampling produces fewer tasks than requested

**Possible causes**:
1. Not enough tasks in easy/hard bins (check logs)
2. Sampling without replacement when pool is too small
3. Benchmark ratios don't sum to 1.0

**Solution**: Check logs for warnings, adjust `sample_with_replacement` or `benchmark_ratios`

### Issue: Import errors

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yaml
```

## 📊 Performance

| Operation                        | Time       | Notes                 |
| -------------------------------- | ---------- | --------------------- |
| Preprocess HumanEval (164 tasks) | ~2-5 sec   | Includes download     |
| Preprocess GSM8K (1,319 tasks)   | ~10-20 sec | Includes download     |
| Preprocess all 5 benchmarks      | ~1-2 min   | One-time operation    |
| Generate exp1 (1000 tasks)       | <1 sec     | From cached full data |
| Generate exp5 (2,000 tasks)      | ~2-3 sec   | From cached full data |

**Disk Space**:
- Full preprocessed data: ~50-100 MB
- Experiment pools: ~0.5-2 MB each
- HuggingFace cache: ~500 MB-1 GB

## 📋 Quick Reference for Researchers

### Most Common Operations

#### 1. Run Default Experiments
```bash
conda activate symphony-data-gen
python src/quick_start.py
```

#### 2. Modify Experiment Parameters
Edit `src/quick_start.py`, find the experiment you want to change:
```python
'exp1': {
    'n_total_tasks': 1000,        # Change number of tasks
    'difficulty_split': '80:20',  # Change easy:hard ratio
    'benchmark_ratios': {...},    # Change benchmark allocation
}
```

#### 3. Enable/Disable Benchmarks
Edit `config/data_config.yaml`:
```yaml
benchmarks:
  humaneval:
    enabled: false  # Disable this benchmark
```

#### 4. Change Difficulty Thresholds
Edit `config/data_config.yaml`:
```yaml
difficulty_percentiles:
  humaneval: [30, 70]  # Make more tasks easy/hard (less medium)
```

#### 5. Read Generated Tasks
```python
import json
with open('data/exp1/task_pool.jsonl') as f:
    for line in f:
        task = json.loads(line)
        # task['benchmark'], task['difficulty_bin'], task['raw_data']
```

### Parameter Cheat Sheet

| Parameter                 | Where                     | Values     | Purpose                       |
| ------------------------- | ------------------------- | ---------- | ----------------------------- |
| `enabled`                 | `config/data_config.yaml` | true/false | Enable/disable benchmark      |
| `difficulty_percentiles`  | `config/data_config.yaml` | [20, 80]   | Define easy/hard thresholds   |
| `sample_with_replacement` | `config/data_config.yaml` | true/false | Allow duplicate tasks         |
| `n_total_tasks`           | `quick_start.py`          | integer    | Total tasks in experiment     |
| `difficulty_split`        | `quick_start.py`          | "80:20"    | Ratio of easy:hard            |
| `benchmark_ratios`        | `quick_start.py`          | {bm: 0.5}  | Task allocation per benchmark |
| `random_seed`             | `quick_start.py`          | integer    | Reproducibility seed          |

### Task Count Limits (Default [20,80] Thresholds)

| Benchmark  | Total | Easy | Hard | Can Use Without Replacement? |
| ---------- | ----- | ---- | ---- | ---------------------------- |
| HumanEval  | 164   | ~33  | ~33  | ⚠️ Limited (up to 33 each)    |
| GSM8K      | 1,319 | ~264 | ~332 | ✅ Yes (plenty available)     |
| BBH        | 2,437 | ~487 | ~487 | ✅ Yes (plenty available)     |
| AMC        | 83    | ~18  | ~18  | ⚠️ Very Limited (up to 18)    |
| Medical QA | 1,273 | ~254 | ~254 | ✅ Yes (plenty available)     |

**Tip**: If you need more tasks from HumanEval or AMC, either:
1. Set `sample_with_replacement: true` in config (allows duplicates)
2. Change thresholds to [30, 70] (gives ~49 easy/hard for HumanEval)

## 📚 Additional Resources

- **Source code**: `src/data_generator.py` contains detailed docstrings
- **Configuration**: `config/data_config.yaml` includes inline comments
- **Quick start**: `src/quick_start.py` demonstrates full workflow

## ❓ FAQ

**Q: How do I add a new experiment?**  
A: Edit `src/quick_start.py`, add a new entry to the `experiments` dict, then run the script.

**Q: Can I use only one benchmark?**  
A: Yes! Set `benchmarks: ['humaneval']` and `benchmark_ratios: {'humaneval': 1.0}`

**Q: Why am I getting fewer tasks than requested?**  
A: Check logs - likely not enough easy/hard tasks available. Either enable `sample_with_replacement` or adjust `difficulty_percentiles`.

**Q: How do I make tasks easier/harder?**  
A: Adjust `difficulty_split` (e.g., '90:10' = mostly easy, '10:90' = mostly hard)

**Q: How do I change what makes a task "easy" or "hard"?**  
A: Modify `difficulty_percentiles` in config. Default [20,80] means bottom 20% = easy, top 20% = hard.

**Q: Can I mix all 5 benchmarks?**  
A: Yes! See exp5 example. Set `benchmarks: ['humaneval', 'gsm8k', 'bbh', 'amc', 'medical_qa']`

**Q: How long does preprocessing take?**  
A: ~1-2 minutes for all 5 benchmarks (one-time). After that, generating experiments is instant (<1 second).