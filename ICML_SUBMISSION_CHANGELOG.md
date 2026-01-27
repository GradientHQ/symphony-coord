# ICML Submission Code Cleanup Changelog

This document tracks all changes made to prepare the codebase for ICML 2026 submission.

## Summary

The codebase has been cleaned up for ICML submission while maintaining full reproducibility. All experiment scripts are preserved; only generated result data files have been removed.

## Changes Made

### 1. Deleted Generated Results (Data Files Only)

The following result directories containing CSV/XLSX/JSON output files were deleted:

- `experiments/result/` - Experiment 4 results
- `experiments/exp3_real/results/` - Real experiment 3 results
- `experiments/exp3/sim/results/` - Simulation experiment 3 results
- `experiments/exp3/real/results/` - Real experiment 3 results
- `experiments/exp1_real_openrouter/results/` - Experiment 1 OpenRouter results
- `experiments/exp1_sim_efficiency_cost/results/` - Experiment 1 simulation results
- `results/` - Top-level results folder
- `logs/` - Runtime logs

**Note:** All experiment Python scripts (*.py) and configuration files (*.yaml) are preserved for reproducibility.

### 2. Deleted Build Artifacts and IDE Files

- `.idea/` - PyCharm IDE configuration
- `symphony.egg-info/` - Python package build metadata
- `repair.js` - Temporary fix script

### 3. Deleted Duplicate Configuration Folders

- `runtime/configs/openrouter/llama-3.1-70b-instruct copy/`
- `runtime/configs/openrouter/openai-gpt-5-mini copy/`

### 4. Deleted Separate Subprojects

- `symphony-data-generator/` - Separate data generation project (not core Symphony)
- `.github/` - GitHub-specific configuration
- `runtime/sim_efficiency_cost/` - Simulation result CSVs
- `runtime/results/` - Runtime results
- `SETUP_PACKAGING.md` - Internal packaging documentation

### 5. Anonymized for Review

**README.md:**
- Removed contact email (tianyu@gradient.network)
- Replaced author names in citation with "Anonymous"
- Removed Discord and Twitter social links
- Updated citation year to 2026

**pyproject.toml:**
- Verified no author information present (clean)

### 6. Updated .gitignore

Added patterns to exclude:
- IDE files (`.idea/`, `.vscode/`)
- Package metadata (`*.egg-info/`)
- Generated results (`**/results/*.csv`, `**/results/*.xlsx`, `**/results/*.json`)
- Log files (`logs/`)

## Files Preserved for Reproducibility

### Core Framework
- `core/` - All 10 core modules (symphony.py, routing.py, linucb_selector.py, etc.)
- `agents/` - Agent and User implementations
- `protocol/` - Task contracts, beacons, responses
- `infra/` - ISEP client, network adapter
- `main.py`, `symphony.py` - Entry points

### Experiment Scripts (All Preserved)
- `Pre-train.py` - Main experiment runner
- `experiments/` - All Python scripts and YAML configs
- `plot_*.py` - All 8 visualization scripts
- `run_*.sh` - All 6 shell scripts for running experiments
- Analysis utilities (analyze_routing.py, recalc_acc.py, etc.)

### Configuration and Documentation
- `runtime/configs/` - Agent configurations (duplicates removed)
- `examples/` - Usage examples
- `tests/` - Test suite
- `tools/` - Utility tools
- `README.md` - Main documentation (anonymized)
- `requirements.txt` - Dependencies
- `LICENSE` - License file

## Security Verification

- No hardcoded API keys found
- All API keys are read from environment variables (os.getenv())
- Documentation only shows placeholder examples (sk-or-v1-...)

## Reproducing Results

To reproduce the experiments:

1. Install dependencies: `pip install -r requirements.txt`
2. Set up API keys: `export OPENROUTER_API_KEY="your-key"`
3. Run experiments using the preserved shell scripts or Python files
4. Results will be generated in the appropriate results directories

---

*Generated for ICML 2026 submission*
