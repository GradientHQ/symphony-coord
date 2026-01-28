# Symphony 2.0 Data Generator

A unified framework for generating experiment-ready task pools with difficulty scoring.

## Quick Start

```bash
cd symphony-data-generator
pip install -r requirements.txt
python src/quick_start.py
```

## Difficulty Scoring Formulas

### HumanEval (Code Generation)
```
d_code = 0.6 * (n_asserts / max_asserts) + 0.4 * (prompt_len / max_prompt_len)
```

### GSM8K (Mathematical Reasoning)
```
d_math = reasoning_steps / max_reasoning_steps
```

### BBH (Multi-hop Reasoning)
```
d_BBH = c_task + 0.3 * (input_len / max_input_len)
```
Where `c_task` is a task-specific base complexity (0.25-0.85).

### AMC (Competition Mathematics)
```
d_AMC = 0.7 * (problem_len / max_problem_len) + 0.3 + 0.12 * has_latex
```
Where `has_latex` indicates presence of LaTeX notation (\ or $).

### Medical QA (Domain-Specific)
```
d_med = 0.4 * question_len + 0.3 * keyword_density + 0.2 * option_len + 0.1 * clinical_case
```

## Normalization

1. **95th percentile normalizers**: Each benchmark uses data-driven normalization constants computed from the 95th percentile of the full dataset.

2. **Min-max normalization**: After raw scoring, all scores are normalized within each benchmark to [0, 1].

3. **Percentile-based binning**: Tasks are binned using P20/P80 thresholds from the FULL dataset:
   - Easy: score ≤ P20
   - Hard: score ≥ P80
   - Medium: P20 < score < P80

## Configuration

Edit `config/data_config.yaml` to:
- Enable/disable benchmarks
- Adjust difficulty percentile thresholds
- Configure sampling behavior
