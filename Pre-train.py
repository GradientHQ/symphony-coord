#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import random
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
	sys.path.insert(0, _THIS_DIR)

import symphony as symphony_module
from agents.agent import Agent
from protocol.task_contract import Task


def load_yaml(path: str) -> Dict[str, Any]:
	try:
		import yaml  # type: ignore
	except Exception as e:
		raise RuntimeError("PyYAML 未安装，无法读取 agent 配置") from e
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def load_tasks(path: str, n: Optional[int], seed: int, benchmark: Optional[str]) -> List[Dict[str, Any]]:
	tasks: List[Dict[str, Any]] = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			if benchmark and obj.get("benchmark") != benchmark:
				continue
			tasks.append(obj)

	if n is None or n <= 0 or n >= len(tasks):
		return tasks

	rng = random.Random(seed)
	return rng.sample(tasks, k=int(n))


def task_to_text(task: Dict[str, Any]) -> str:
	raw = task.get("raw_data") or {}
	return (
		raw.get("prompt")
		or raw.get("question")
		or raw.get("input")
		or task.get("prompt")
		or task.get("input")
		or task.get("text")
		or ""
	)


def build_task_obj(task: Dict[str, Any], i: int) -> Task:
	task_text = task_to_text(task)
	return Task.from_dict(
		{
			"task_id": str(task.get("task_id") or task.get("id") or f"task_{i}"),
			"description": task_text,
			"requirements": ["general-reasoning"],
			"context": {
				"benchmark": task.get("benchmark", ""),
				"difficulty_bin": task.get("difficulty_bin", ""),
			},
		}
	)


def is_success(text: str) -> int:
	if not text:
		return 0
	s = text.strip()
	if not s:
		return 0
	if s.startswith("[ERROR]") or s.startswith("[AGENT_ERROR]"):
		return 0
	return 1


def _normalize_answer(x: Any) -> str:
	if x is None:
		return ""
	if isinstance(x, (int, float)):
		return str(x).strip().lower()
	s = str(x).strip().lower()
	for ch in ["\n", "\t", ",", ".", ":", ";", "!", "?", "\"", "'", "(", ")", "[", "]"]:
		s = s.replace(ch, " ")
	return " ".join(s.split())


def _canonical_num(num_str: str) -> str:
	try:
		val = float(num_str)
	except Exception:
		return num_str.strip()
	if abs(val - round(val)) < 1e-9:
		return str(int(round(val)))
	text = f"{val:.10f}".rstrip("0").rstrip(".")
	return text


def _extract_last_number(text: str) -> str:
	import re
	if not text:
		return ""
	clean = text.replace(",", "").replace("$", "").replace("%", "")
	nums = re.findall(r"-?\d+(?:\.\d+)?", clean)
	return _canonical_num(nums[-1]) if nums else ""


def _try_json_final_answer(text: str) -> str:
	if not text:
		return ""
	try:
		obj = json.loads(text)
		if isinstance(obj, dict):
			for key in ["final_answer", "answer", "final", "output"]:
				if key in obj and obj[key] is not None:
					return str(obj[key]).strip()
	except Exception:
		pass
	return ""


def extract_pred(text: str) -> str:
	if not text:
		return ""
	json_ans = _try_json_final_answer(text)
	if json_ans:
		return json_ans
	low = text.lower()
	key = "final answer:"
	if key in low:
		idx = low.rfind(key)
		return text[idx + len(key):].strip()
	if "####" in text:
		return text.split("####")[-1].strip()
	return text.strip()


def extract_gold_text(task: Dict[str, Any]) -> str:
	raw = task.get("raw_data") or {}
	bench = str(task.get("benchmark", "")).strip().lower()
	if bench == "bbh":
		return str(raw.get("target", ""))
	return str(raw.get("answer", ""))


def is_correct(pred: str, gold_text: str, benchmark: str) -> int:
	bench = (benchmark or "").strip().lower()
	if bench in {"gsm8k", "gsm"}:
		pnum = _extract_last_number(pred)
		gnum = _extract_last_number(gold_text)
		if pnum and gnum:
			return 1 if _canonical_num(pnum) == _canonical_num(gnum) else 0
	return 1 if _normalize_answer(pred) and _normalize_answer(pred) == _normalize_answer(gold_text) else 0


def load_agents_from_runtime(config_dir: str, agent_ids: List[int]) -> List[Agent]:
	agents: List[Agent] = []
	for aid in agent_ids:
		path = os.path.join(config_dir, f"config_agent_openrouter_{aid}.yaml")
		if not os.path.exists(path):
			raise FileNotFoundError(f"Agent config not found: {path}")
		cfg = load_yaml(path)
		agents.append(Agent(config=cfg))
	return agents


class FrozenSelector:
	def __init__(self, base):
		self._base = base

	def select(self, candidates):
		return self._base.select(candidates)

	def update(self, x, reward):
		return None

	def __getattr__(self, item):
		return getattr(self._base, item)


def save_selector(selector: Any, path: str) -> None:
	state = {
		"d": int(getattr(selector, "d", 0)),
		"l2": float(getattr(selector, "l2", 1.0)),
		"alpha": float(getattr(selector, "alpha", 1.0)),
		"delta": float(getattr(selector, "delta", 0.05)),
		"S": float(getattr(selector, "S", 1.0)),
		"t": int(getattr(selector, "t", 0)),
		"A_inv": getattr(selector, "A_inv", []),
		"b": getattr(selector, "b", []),
	}
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		f.write(json.dumps(state, ensure_ascii=False))


def load_selector(path: str):
	with open(path, "r", encoding="utf-8") as f:
		state = json.loads(f.read())
	sel = symphony_module.GlobalLinUCB(
		d=int(state.get("d", 6)),
		l2=float(state.get("l2", 1.0)),
		alpha=float(state.get("alpha", 1.0)),
		delta=float(state.get("delta", 0.05)),
		S=float(state.get("S", 1.0)),
	)
	sel.A_inv = state.get("A_inv", sel.A_inv)
	sel.b = state.get("b", sel.b)
	sel.t = int(state.get("t", 0))
	return sel


def run_phase(
	phase: str,
	tasks: List[Dict[str, Any]],
	start_index: int,
	cot_count: int,
	outdir: str,
	verbose: bool,
	print_each_step: bool,
) -> Tuple[int, List[Dict[str, Any]]]:
	logs: List[Dict[str, Any]] = []
	for i, task in enumerate(tasks, start=start_index):
		t0 = time.time()
		task_obj = build_task_obj(task, i=i)
		trace = symphony_module.execute_task(task_obj, cot_count=cot_count, return_mode="trace")
		final_text = ""
		agent_ids: List[str] = []
		if isinstance(trace, dict):
			traces = trace.get("traces", {}) or {}
			# ✅ DEBUG: Print trace structure for first task to diagnose ok=0 issue
			if i == start_index:
				print(f"\n[DEBUG] === Trace structure for first task (step={i}) ===")
				print(f"[DEBUG] trace type: {type(trace)}, is_dict: {isinstance(trace, dict)}")
				print(f"[DEBUG] traces type: {type(traces)}, is_dict: {isinstance(traces, dict)}, len: {len(traces) if isinstance(traces, dict) else 0}")
				if traces:
					first = next(iter(traces.values()))
					print(f"[DEBUG] first type: {type(first)}, is_dict: {isinstance(first, dict)}")
					if isinstance(first, dict):
						print(f"[DEBUG] first keys: {list(first.keys())}")
						print(f"[DEBUG] voted: {repr(first.get('voted', 'MISSING'))[:200]}")
						print(f"[DEBUG] voted_final: {repr(first.get('voted_final', 'MISSING'))[:200]}")
						print(f"[DEBUG] runs type: {type(first.get('runs'))}, len: {len(first.get('runs', []))}")
						if first.get("runs"):
							first_run = first["runs"][0]
							print(f"[DEBUG] first_run type: {type(first_run)}, keys: {list(first_run.keys()) if isinstance(first_run, dict) else 'NOT_DICT'}")
							if isinstance(first_run, dict):
								print(f"[DEBUG] first_run agent_id: {repr(first_run.get('agent_id', 'MISSING'))}")
				else:
					print(f"[DEBUG] WARNING: traces is empty!")
				print(f"[DEBUG] ============================================\n")
			if traces:
				first = next(iter(traces.values()))
				if isinstance(first, dict):
					final_text = str(first.get("voted", "") or "")
					runs = first.get("runs", []) or []
					for r in runs:
						if isinstance(r, dict):
							aid = r.get("agent_id")
							if aid:
								agent_ids.append(str(aid))
		dt = time.time() - t0
		ok = is_success(final_text)
		pred = extract_pred(final_text)
		gold_text = extract_gold_text(task)
		acc = is_correct(pred, gold_text, str(task.get("benchmark", "")))
		logs.append(
			{
				"i": i,
				"phase": phase,
				"task_id": task.get("task_id") or task.get("id"),
				"benchmark": task.get("benchmark"),
				"difficulty_bin": task.get("difficulty_bin"),
				"agent_ids": agent_ids,
				"ok": ok,
				"pred": pred,
				"gold": gold_text,
				"acc": acc,
				"latency_s": dt,
			}
		)
		if print_each_step:
			first_agent = agent_ids[0] if agent_ids else "NA"
			print(f"[{phase}] step={i} agent={first_agent} ok={ok} acc={acc} latency={dt:.2f}s")
		elif verbose and (i % 10 == 0):
			print(f"[{phase}] {i} done, ok={ok}, acc={acc}, latency={dt:.2f}s")

	os.makedirs(outdir, exist_ok=True)
	log_path = os.path.join(outdir, f"pretrain_{phase}.jsonl")
	with open(log_path, "w", encoding="utf-8") as f:
		for r in logs:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")

	return start_index + len(tasks), logs


def plot_acc_curve(logs: List[Dict[str, Any]], outdir: str) -> None:
	try:
		import matplotlib.pyplot as plt
	except Exception as e:
		print(f"[WARN] matplotlib not available, skip acc plot: {e}")
		return

	if not logs:
		return

	logs_sorted = sorted(logs, key=lambda x: int(x.get("i", 0)))
	steps = [int(x.get("i", 0)) for x in logs_sorted]
	ok_vals = [int(x.get("acc", x.get("ok", 0))) for x in logs_sorted]

	cum = []
	count = 0
	for i, v in enumerate(ok_vals, start=1):
		count += int(v)
		cum.append(count / i)

	plt.figure(figsize=(9.5, 4.2), dpi=160)
	plt.plot(steps, cum, label="Cumulative ACC", linewidth=1.6)
	plt.ylim(0.0, 1.02)
	plt.xlabel("Step")
	plt.ylabel("ACC")
	plt.title("Pre-train ACC Curve")
	plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
	plt.legend(frameon=False, fontsize=9)
	os.makedirs(outdir, exist_ok=True)
	plt.savefig(os.path.join(outdir, "acc_curve.png"))
	plt.close()


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--task-pool", type=str, required=True, help="Path to JSONL task pool")
	ap.add_argument("--benchmark", type=str, default=None, help="Filter tasks by benchmark (e.g., bbh)")
	ap.add_argument("--n", type=int, default=None, help="Total tasks to use")
	ap.add_argument("--seed", type=int, default=123)
	ap.add_argument("--cold-n", type=int, default=50)
	ap.add_argument("--pretrain-n", type=int, default=200)
	ap.add_argument("--val-n", type=int, default=250)
	ap.add_argument("--topL", type=int, default=3)
	ap.add_argument("--ucb-alpha", type=float, default=1.0)
	ap.add_argument("--ucb-l2", type=float, default=1.0)
	ap.add_argument("--cot-count", type=int, default=1)
	ap.add_argument("--plan-k", type=int, default=1)
	ap.add_argument("--multi-plan", action="store_true", help="Use all agents as planners (plan_k = agent count)")
	ap.add_argument("--agents", type=str, default="11,12,13,14,15",
	                help="Comma-separated agent IDs to load from runtime-dir (default: 11,12,13,14,15)")
	ap.add_argument("--outdir", type=str, default="pretrain_results")
	ap.add_argument("--runtime-dir", type=str, default="runtime")
	ap.add_argument("--save-selector", type=str, default=None, help="Save UCB state after pretrain")
	ap.add_argument("--load-selector", type=str, default=None, help="Load UCB state and run validation only")
	ap.add_argument("--plot-acc", action="store_true", help="Plot cumulative ACC curve")
	ap.add_argument("--print-each-step", action="store_true", help="Print agent per step")
	ap.add_argument("--verbose", action="store_true")
	args = ap.parse_args()
	# 默认每步打印，便于观察 agent 是否在响应
	if not args.print_each_step:
		args.print_each_step = True

	tasks = load_tasks(args.task_pool, args.n, seed=args.seed, benchmark=args.benchmark)

	if args.load_selector:
		args.cold_n = 0
		args.pretrain_n = 0

	total_needed = args.cold_n + args.pretrain_n + args.val_n
	if total_needed > len(tasks):
		args.val_n = max(0, len(tasks) - args.cold_n - args.pretrain_n)

	cold_tasks = tasks[: args.cold_n]
	pretrain_tasks = tasks[args.cold_n : args.cold_n + args.pretrain_n]
	val_tasks = tasks[args.cold_n + args.pretrain_n : args.cold_n + args.pretrain_n + args.val_n]

	# load agents from command-line argument
	agent_ids = [int(x.strip()) for x in args.agents.split(",") if x.strip()]
	agents = load_agents_from_runtime(args.runtime_dir, agent_ids)
	for ag in agents:
		symphony_module.register_agent(ag)
	if args.multi_plan:
		args.plan_k = max(1, len(agents))

	idx = 1
	all_logs: List[Dict[str, Any]] = []
	if args.load_selector:
		selector = load_selector(args.load_selector)
		symphony_module._global_orchestrator.selector = FrozenSelector(selector)
		symphony_module._global_orchestrator.use_dynamic = True
		if val_tasks:
			idx, logs = run_phase("validation", val_tasks, idx, args.cot_count, args.outdir, args.verbose, args.print_each_step)
			all_logs.extend(logs)
	else:
		# cold start: static Top-L
		symphony_module.init(
			use_dynamic=False,
			topL=int(args.topL),
			plan_k=int(args.plan_k),
		)
		if cold_tasks:
			idx, logs = run_phase("cold_start", cold_tasks, idx, args.cot_count, args.outdir, args.verbose, args.print_each_step)
			all_logs.extend(logs)

		# pretrain: Top-L + UCB (updates enabled)
		symphony_module.init(
			use_dynamic=True,
			topL=int(args.topL),
			linucb_alpha=float(args.ucb_alpha),
			linucb_l2=float(args.ucb_l2),
			plan_k=int(args.plan_k),
		)
		if pretrain_tasks:
			idx, logs = run_phase("pretrain", pretrain_tasks, idx, args.cot_count, args.outdir, args.verbose, args.print_each_step)
			all_logs.extend(logs)

		selector = symphony_module._global_orchestrator.selector
		if selector is None:
			raise RuntimeError("UCB selector is not initialized; cannot freeze for validation")
		if args.save_selector:
			save_selector(selector, args.save_selector)

		symphony_module._global_orchestrator.selector = FrozenSelector(selector)
		symphony_module._global_orchestrator.use_dynamic = True

		if val_tasks:
			idx, logs = run_phase("validation", val_tasks, idx, args.cot_count, args.outdir, args.verbose, args.print_each_step)
			all_logs.extend(logs)

	if args.plot_acc:
		plot_acc_curve(all_logs, args.outdir)

	print(f"[OK] Pre-train done. outdir={args.outdir}")


if __name__ == "__main__":
	main()
