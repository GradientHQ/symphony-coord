# Symphony-Coord

<h3 align="center">
Emergent Coordination in Decentralized Agent Systems
</h3>

<p align="center">
  <img src="assets/teaser.png" width="92%">
</p>

<p align="center">
  <b>Adaptive Multi-Agent Routing via Online Bandit Coordination</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.00966">📄 Paper</a>
  ·
  <a href="#quick-start">⚡ Quick Start</a>
  ·
  <a href="#reproducing-results">📊 Reproducibility</a>
  ·
  <a href="#project-demo">🎬 Demo</a>
</p>

---

## TL;DR

Symphony-Coord is a decentralized multi-agent coordination framework that formulates adaptive agent routing as an online multi-armed bandit problem.

Instead of relying on fixed orchestration heuristics or centralized planners, Symphony-Coord enables agents to dynamically specialize through online interaction, adaptive routing, and reward-driven coordination.

The framework combines:

* task decomposition
* LinUCB-based routing
* decentralized capability matching
* parallel Chain-of-Thought execution
* voting-based aggregation

into a unified coordination pipeline for robust multi-agent reasoning.

---

# Why Symphony-Coord?

Modern multi-agent systems face several major limitations:

* centralized orchestrators become bottlenecks at scale
* fixed routing heuristics fail under dynamic workloads
* static agent assignment prevents specialization
* coordination robustness degrades under agent failures

Existing orchestration pipelines often assume:

* stable agent behavior
* fixed routing strategies
* homogeneous execution environments

However, real-world decentralized systems are inherently dynamic.

Agent quality, latency, availability, and specialization evolve continuously over time.

Symphony-Coord addresses this challenge by treating routing as an online decision-making problem under uncertainty.

---

# Main Contributions

* **Decentralized Coordination**

  * removes dependence on centralized orchestration
  * enables scalable multi-agent interaction

* **Adaptive Routing via LinUCB**

  * formulates agent selection as an online contextual bandit problem
  * continuously updates routing decisions using reward feedback

* **Emergent Specialization**

  * agents gradually specialize through interaction rather than predefined roles

* **Robust Multi-Path Reasoning**

  * combines parallel Chain-of-Thought execution with voting aggregation

* **Research-Grade Evaluation Pipeline**

  * supports simulation and real-model evaluation across multiple reasoning benchmarks

---

# Main Results

## Efficiency and Robustness Overview

| Method             | Accuracy ↑ | Cost ↓   | Recovery Speed ↑ |
| ------------------ | ---------- | -------- | ---------------- |
| Static Routing     | xx.x       | xx.x     | xx               |
| Random Routing     | xx.x       | xx.x     | xx               |
| Rule-Based Routing | xx.x       | xx.x     | xx               |
| Symphony-Coord     | **xx.x**   | **xx.x** | **xx**           |

---

## Benchmark Highlights

### GSM8K

| Method         | Accuracy |
| -------------- | -------- |
| Baseline       | xx.x     |
| Symphony-Coord | **xx.x** |

### BBH

| Method         | Macro Average |
| -------------- | ------------- |
| Baseline       | xx.x          |
| Symphony-Coord | **xx.x**      |

### Robustness Recovery

| Scenario          | Recovery Tasks |
| ----------------- | -------------- |
| Agent Failure     | xx             |
| Agent Degradation | xx             |

> Replace placeholder numbers with final experimental results.

---

# Method Overview

<p align="center">
  <img src="assets/pipeline.png" width="95%">
</p>

Symphony-Coord follows a three-stage coordination pipeline.

---

## 1. Planning Phase

Multiple planning agents decompose complex user queries into executable sub-tasks.

The system evaluates candidate plans using contextual reward estimation.

### Core Components

* task decomposition
* plan proposal generation
* LinUCB plan selection

---

## 2. Execution Phase

Each sub-task is broadcast through decentralized beacon routing.

Agents are dynamically selected based on:

* capability matching
* historical reward feedback
* online uncertainty estimation

Selected agents then execute reasoning chains in parallel.

### Core Components

* beacon broadcasting
* capability matching
* contextual bandit routing
* parallel CoT execution

---

## 3. Voting Phase

The framework aggregates multiple reasoning paths using voting-based response fusion.

This improves:

* robustness
* answer consistency
* fault tolerance

### Core Components

* CoT voting
* response aggregation
* confidence estimation

---

# Architecture

```text
User Query
    │
    ▼
┌─────────────────────────────────────┐
│ Planning Phase                      │
│ - Task decomposition                │
│ - Candidate plan generation         │
│ - LinUCB plan selection             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Execution Phase                     │
│ - Beacon broadcasting               │
│ - Capability matching               │
│ - Online bandit routing             │
│ - Parallel CoT execution            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Voting Phase                        │
│ - Multi-response aggregation        │
│ - Confidence voting                 │
│ - Final answer generation           │
└─────────────────────────────────────┘
    │
    ▼
Final Response
```

---

# Key Features

## Decentralized Coordination

No centralized orchestration bottleneck.

Agents interact through distributed coordination protocols.

---

## Adaptive Online Routing

Routing policies continuously evolve through reward-driven learning.

---

## Emergent Specialization

Agents dynamically specialize according to observed task performance.

---

## Robust Multi-Agent Reasoning

Parallel CoT execution improves reasoning robustness and fault tolerance.

---

## Edge-Friendly Deployment

Supports consumer-grade GPUs and heterogeneous execution environments.

---

# Project Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=Qnh4lrXGprE">
    <img src="https://img.youtube.com/vi/Qnh4lrXGprE/maxresdefault.jpg" width="900">
  </a>
</p>

---

# Quick Start

## Clone Repository

```bash
git clone https://github.com/GradientHQ/symphony-coord.git
cd symphony-coord
```

---

## Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Configure API Key

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key"
```

---

## Run Simple Example

```python
from symphony import SymphonyOrchestrator

orchestrator = SymphonyOrchestrator(
    agents=["agent1", "agent2", "agent3"],
    topL=3,
    cot_count=3
)

result = orchestrator.run_task(
    task_description="Solve: What is 25 * 37?",
    requirements=["math"]
)

print(result["final_answer"])
```

---

# Repository Structure

```text
symphony-coord/
├── agents/                    # Agent implementations
├── core/                      # Core routing and coordination algorithms
├── experiments/               # Benchmark experiments
├── infra/                     # Infrastructure and protocols
├── models/                    # Model loading utilities
├── protocol/                  # Communication protocol definitions
├── scripts/                   # Visualization and analysis scripts
├── tests/                     # Unit tests
├── docs/                      # Documentation
└── symphony.py                # Main orchestrator
```

---

# Experiments

## Experiment 1 — Efficiency & Cost

Evaluates:

* routing efficiency
* API cost reduction
* adaptive selection quality

### Run

```bash
python experiments/exp1/real/exp1_real_openrouter.py --n 100
```

---

## Experiment 2 — Robustness & Recovery

Evaluates:

* adaptation under agent failure
* recovery after degradation
* coordination stability

### Run

```bash
bash experiments/exp2/scripts/run_exp2_both.sh
```

---

## Experiment 3 — System Optimization

Evaluates:

* latency balancing
* heterogeneous execution
* load-aware routing

### Run

```bash
bash experiments/exp3/run_exp3.sh
```

---

# Visualization

## Suggested README Assets

Recommended figures/GIFs:

* routing visualization
* specialization heatmaps
* recovery curves
* execution timelines
* decentralized coordination diagrams

Suggested directory:

```text
assets/
├── teaser.png
├── pipeline.png
├── routing_demo.gif
├── robustness_curve.png
└── specialization_heatmap.png
```

---

# Reproducing Results

## Full Benchmark Evaluation

```bash
bash experiments/scripts/run_all_datasets.sh
```

---

## Generate Paper Figures

```bash
python scripts/plotting/paper_figures/plot_robustness_bars.py
python scripts/plotting/paper_figures/plot_gap_analysis.py
```

---

## Expected Outputs

```text
pretrain_results/
├── accuracy_summary.csv
├── progress_state.json
├── selection_trace.json
└── routing_visualizations/
```

---

# Documentation

Detailed documentation has been moved into the `docs/` directory.

## Available Docs

```text
docs/
├── INSTALL.md
├── EXPERIMENTS.md
├── CONFIGS.md
├── TROUBLESHOOTING.md
└── OPENROUTER_CONFIG_GUIDE.md
```

---

# System Requirements

| Requirement | Minimum             | Recommended   |
| ----------- | ------------------- | ------------- |
| Python      | 3.9                 | 3.10 / 3.11   |
| RAM         | 8 GB                | 16 GB         |
| GPU         | Optional            | RTX 3060+     |
| OS          | Linux/macOS/Windows | Ubuntu 20.04+ |

---

# Citation

```bibtex
@article{guan2026symphony,
  title={Symphony-Coord: Emergent Coordination in Decentralized Agent Systems},
  author={Guan, Zhaoyang and Cao, Huixi and Zhong, Ming and Yang, Eric and Ai, Lynn and Ni, Yongxin and Shi, Bill},
  journal={arXiv preprint arXiv:2602.00966},
  year={2026}
}
```

---

# Acknowledgements

We thank the open-source research community for foundational work in:

* decentralized systems
* online bandit optimization
* multi-agent reasoning
* Chain-of-Thought coordination
* distributed inference systems

---

# License

This project is released under the MIT License.
