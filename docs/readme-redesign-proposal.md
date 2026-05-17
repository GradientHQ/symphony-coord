<h1 align="center">
Symphony-Coord
</h1>

<h3 align="center">
Agents That Learn Who Should Solve What
</h3>

<p align="center">
Self-Organizing Multi-Agent Coordination via Adaptive Online Routing
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.00966">
    <img src="https://img.shields.io/badge/arXiv-2602.00966-b31b1b">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue">
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.00966">📄 Paper</a>
  ·
  <a href="https://symphonycoord.ai">🌐 Live Demo</a>
  ·
  <a href="https://github.com/GradientHQ">💡 Ecosystem</a>
</p>

---

<p align="center">
  <a href="https://symphonycoord.ai">
    <img src="assets/hero_demo.png" width="94%">
  </a>
</p>

<p align="center">
Decentralized agents that dynamically learn who should solve what.
</p>

---

# Overview

Symphony-Coord is a decentralized multi-agent coordination framework where agents dynamically learn:

- who should solve what
- when to route tasks
- how to specialize through interaction

Instead of relying on fixed orchestration heuristics or centralized planners, Symphony-Coord formulates routing as an online decision-making problem under uncertainty.

Routing policies continuously evolve through:

- contextual online routing
- reward-driven adaptation
- decentralized coordination
- emergent specialization

The framework is designed for dynamic environments where:

- agent capability changes over time
- latency fluctuates
- nodes fail or degrade
- specialization must emerge online

---

# Why Symphony-Coord?

Modern multi-agent systems often rely on:

- centralized orchestrators
- static expert assignment
- fixed routing heuristics

However, real-world decentralized systems are inherently dynamic.

Agent capability, latency, availability, and specialization continuously evolve during execution.

Symphony-Coord studies how robust coordination and specialization can emerge through online interaction instead of predefined orchestration rules.

---

# System Demo

Explore adaptive routing and emergent specialization in decentralized multi-agent systems.

<p align="center">
  <a href="https://www.youtube.com/watch?v=Qnh4lrXGprE">
    <img src="https://img.youtube.com/vi/Qnh4lrXGprE/maxresdefault.jpg" width="900">
  </a>
</p>

---

# Dynamic System Behavior

## Adaptive Routing Evolution

<p align="center">
  <img src="assets/routing_demo.gif" width="92%">
</p>

Routing decisions evolve online as task streams and reward feedback continuously change.

---

## Emergent Agent Specialization

<p align="center">
  <img src="assets/specialization_heatmap.png" width="84%">
</p>

Agents gradually specialize through interaction and reward feedback instead of predefined static roles.

---

## Robust Failure Recovery

<p align="center">
  <img src="assets/recovery_curve.png" width="84%">
</p>

The system dynamically adapts after agent degradation, routing disruption, or node failure.

---

# Main Results

| Evaluation Setting | Improvement |
|---|---:|
| Routing Cost vs. Static Routing | ↓ 23% |
| Recovery Speed under Agent Failure | ↑ 2.1× |
| GSM8K Accuracy vs. Routing Baseline | ↑ 8.4% |

> Evaluated across GSM8K, BBH, robustness recovery, and heterogeneous system optimization benchmarks.

---

# Interactive Demo

Explore adaptive routing and emergent specialization in real time.

<p align="center">
  <a href="https://symphonycoord.ai">
    <img src="assets/frontend_preview.png" width="92%">
  </a>
</p>

### Interactive Features

- live routing visualization
- evolving specialization dynamics
- decentralized coordination simulation
- adaptive recovery under failure
- multi-agent execution tracing

---

# Core Features

## 🧠 Emergent Specialization

Agents dynamically specialize through online interaction and reward feedback.

No predefined expert assignment is required.

---

## ⚡ Adaptive Online Routing

Routing decisions continuously evolve using contextual bandit optimization and online reward estimation.

---

## 🌐 Decentralized Coordination

No centralized orchestration bottleneck.

Agents coordinate through distributed routing and capability-aware interaction.

---

## 🔄 Robust Failure Recovery

The framework adapts under:

- unavailable agents
- degraded performance
- latency shifts
- dynamic workloads

---

## 🚀 Parallel Multi-Path Reasoning

Symphony-Coord combines:

- decentralized routing
- parallel Chain-of-Thought execution
- voting-based aggregation

for robust multi-agent reasoning.

---

# System Architecture

<p align="center">
  <img src="assets/pipeline.png" width="92%">
</p>

Symphony-Coord follows a three-stage coordination pipeline.

---

## 1. Planning

🧩 Task decomposition and candidate plan generation.

Core components:

- task decomposition
- plan proposal generation
- uncertainty-aware plan selection

---

## 2. Adaptive Routing

🌐 Decentralized capability-aware coordination.

Core components:

- contextual routing
- capability matching
- online reward adaptation
- emergent specialization

---

## 3. Voting & Aggregation

🧠 Robust multi-path reasoning fusion.

Core components:

- parallel CoT execution
- confidence estimation
- voting-based aggregation
- final answer fusion

---

# Quick Start

## Installation

```bash
git clone https://github.com/GradientHQ/symphony-coord.git
cd symphony-coord

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
````

---

## Configure API Key

```bash
export OPENROUTER_API_KEY="your-key"
```

---

## Run Example

```python
from symphony import SymphonyOrchestrator

orchestrator = SymphonyOrchestrator(
    agents=["agent1", "agent2", "agent3"],
    topL=3,
    cot_count=3,
)

result = orchestrator.run_task(
    task_description="Solve: What is 25 * 37?",
    requirements=["math"],
)

print(result["final_answer"])
```

---

# Reproducing Results

Run the benchmark suite:

```bash
bash experiments/scripts/run_all_datasets.sh
```

Generate paper figures:

```bash
python scripts/plotting/paper_figures/plot_robustness_bars.py
python scripts/plotting/paper_figures/plot_gap_analysis.py
```

---

# Ecosystem

Explore the Symphony-Coord ecosystem.

### Resources

* 🌐 Interactive system demo
* 💡 Research discussions
* 📈 Routing and specialization visualization
* 🛠 Open experiments and extensions

### Links

* [GradientHQ](https://github.com/GradientHQ)
* [GitHub Discussions](https://github.com/GradientHQ/symphony-coord/discussions)
* [Issues](https://github.com/GradientHQ/symphony-coord/issues)

---

# Roadmap

* [ ] Interactive routing visualization
* [ ] Dynamic specialization analysis
* [ ] Multi-node distributed deployment
* [ ] Real-time coordination dashboard
* [ ] Open benchmark suite
* [ ] Agent memory and long-horizon coordination

---

# Documentation

Detailed setup and experiment guides are available in:

```text
docs/
├── INSTALL.md
├── EXPERIMENTS.md
├── CONFIGS.md
├── TROUBLESHOOTING.md
└── OPENROUTER_CONFIG_GUIDE.md
```

---

# Repository Structure

```text
symphony-coord/
├── agents/                    # Agent implementations
├── core/                      # Routing and coordination algorithms
├── experiments/               # Benchmark and robustness experiments
├── protocol/                  # Task and beacon protocols
├── scripts/                   # Plotting and analysis scripts
├── docs/                      # Documentation
├── tests/                     # Test suite
└── symphony.py                # Main orchestrator
```

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

MIT License
