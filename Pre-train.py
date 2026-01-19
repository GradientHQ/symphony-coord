#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import sys
import json
import time
import datetime
import argparse
import csv
import signal
import random
import difflib
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


def _normalize_requirements(reqs: Optional[List[str]]) -> List[str]:
	if not reqs:
		return ["analysis"]
	# filter empty and normalize to strings
	cleaned = [str(r).strip() for r in reqs if str(r).strip()]
	return cleaned if cleaned else ["analysis"]


def build_task_obj(task: Dict[str, Any], i: int, requirements: Optional[List[str]] = None) -> Task:
	task_text = task_to_text(task)
	bench = str(task.get("benchmark", "")).strip().lower()
	if bench == "medical_qa":
		raw = task.get("raw_data") or {}
		opts = raw.get("options") or {}
		if isinstance(opts, dict) and opts:
			ordered = []
			for k in ["A", "B", "C", "D", "E"]:
				if k in opts:
					ordered.append(f"{k}. {opts[k]}")
			if ordered:
				task_text = task_text + "\n\n[OPTIONS]\n" + "\n".join(ordered)
		# Force MCQ token output to avoid free-form answers causing acc=0
		task_text = (
			task_text
			+ "\n\n[ANSWER_FORMAT]\n"
			+ "ANSWER_FORMAT: MCQ_TOKEN\n"
			+ "ALLOWED_TOKENS: A,B,C,D,E\n"
			+ "Return ONLY the single token (e.g., A). No extra words."
		)
		# Previous behavior (kept for reference):
		# task_text = task_text  # free-form answer text (may cause acc=0 for medical_qa)
	task_reqs = requirements
	if task_reqs is None:
		# Prefer task-provided requirements if present
		task_reqs = task.get("requirements") if isinstance(task, dict) else None
	task_reqs = _normalize_requirements(task_reqs)
	return Task.from_dict(
		{
			"task_id": str(task.get("task_id") or task.get("id") or f"task_{i}"),
			"description": task_text,
			"requirements": task_reqs,
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
	t = text.strip()

	# 1) strip ```json ... ``` fences if present
	if t.startswith("```"):
		lines = t.splitlines()
		# remove first fence line (```json or ```)
		if lines and lines[0].startswith("```"):
			lines = lines[1:]
		# remove last fence line if it is ```
		if lines and lines[-1].strip() == "```":
			lines = lines[:-1]
		t = "\n".join(lines).strip()

	# 2) try parse as JSON
	try:
		obj = json.loads(t)
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
	if bench == "humaneval":
		return str(
			raw.get("canonical_solution")
			or raw.get("solution")
			or raw.get("reference")
			or raw.get("answer", "")
		)
	return str(raw.get("answer", ""))


def _clean_code_block(text: str) -> str:
	if not text:
		return ""
	s = text.strip()
	if s.startswith("```"):
		lines = s.splitlines()
		if lines and lines[0].startswith("```"):
			lines = lines[1:]
		if lines and lines[-1].startswith("```"):
			lines = lines[:-1]
		return "\n".join(lines).strip()
	return text


def _exec_with_timeout(code: str, timeout_s: int = 6) -> None:
	def handler(_signum, _frame):
		raise TimeoutError("humaneval timeout")

	old_handler = None
	if hasattr(signal, "SIGALRM"):
		old_handler = signal.signal(signal.SIGALRM, handler)
		signal.alarm(max(1, int(timeout_s)))
	try:
		globals_dict = {"__builtins__": __builtins__}
		locals_dict: Dict[str, Any] = {}
		exec(code, globals_dict, locals_dict)
	finally:
		if hasattr(signal, "SIGALRM"):
			signal.alarm(0)
			if old_handler is not None:
				signal.signal(signal.SIGALRM, old_handler)


def _format_humaneval_code(prompt: str, pred: str, entry: str) -> str:
	code = _clean_code_block(pred)
	if not code:
		return ""
	lines = code.splitlines()
	has_def = any(line.startswith("def ") for line in lines)
	if has_def:
		return "\n".join([prompt, code])
	# Treat as function body completion: indent under the given prompt signature
	indented = "\n".join([("    " + line) if line.strip() else line for line in lines])
	return "\n".join([prompt, indented])


def _eval_humaneval(pred: str, task: Optional[Dict[str, Any]]) -> int:
	if not task:
		return 0
	raw = task.get("raw_data") or {}
	prompt = str(raw.get("prompt") or "")
	test = str(raw.get("test") or "")
	entry = str(raw.get("entry_point") or "")
	code = _format_humaneval_code(prompt, pred, entry)
	if not (prompt and test and entry and code):
		return 0
	full = "\n".join([code, test, f"check({entry})"])
	try:
		_exec_with_timeout(full, timeout_s=8)
	except Exception:
		return 0
	return 1


def _normalize_mcq_token(s: str) -> str:
	if not s:
		return ""
	s = s.strip().upper()
	if s.startswith("(") and s.endswith(")") and len(s) == 3:
		s = s[1:2]
	return s


def _extract_mcq_token(pred: str) -> str:
	if not pred:
		return ""
	p = pred.strip()
	if not p:
		return ""
	# Prefer explicit leading token
	head = p.split()[0]
	token = _normalize_mcq_token(head)
	if token in {"A", "B", "C", "D", "E"}:
		return token
	# Look for standalone (A)/(B)/...
	for ch in ["A", "B", "C", "D", "E"]:
		if f"({ch})" in p.upper():
			return ch
	return ""


def _map_medical_qa_pred_to_token(task: Optional[Dict[str, Any]], pred: str) -> str:
	if not task or not pred:
		return ""
	# If already a token, keep it
	token = _extract_mcq_token(pred)
	if token:
		return token
	raw = task.get("raw_data") or {}
	opts = raw.get("options") or {}
	if not isinstance(opts, dict) or not opts:
		return ""
	norm_pred = _normalize_answer(pred)
	if not norm_pred:
		return ""
	best_key = ""
	best_score = 0.0
	for k, v in opts.items():
		key = str(k).strip().upper()
		if key not in {"A", "B", "C", "D", "E"}:
			continue
		norm_opt = _normalize_answer(v)
		if not norm_opt:
			continue
		# Exact normalized match
		if norm_opt == norm_pred:
			return key
		# Containment match (short answer inside long option or vice versa)
		if norm_pred in norm_opt or norm_opt in norm_pred:
			score = 0.85
		else:
			score = difflib.SequenceMatcher(None, norm_pred, norm_opt).ratio()
		if score > best_score:
			best_score = score
			best_key = key
	# Prefer confident fuzzy match; otherwise still return best option to enforce A/B/C/D/E output
	if best_key and best_score >= 0.75:
		return best_key
	return best_key or ""


def is_correct(pred: str, gold_text: str, benchmark: str, task: Optional[Dict[str, Any]] = None) -> int:
	bench = (benchmark or "").strip().lower()
	if bench in {"humaneval", "human_eval"}:
		return _eval_humaneval(pred, task)
	if bench in {"medical_qa"}:
		raw = (task or {}).get("raw_data") or {}
		answer_idx = str(raw.get("answer_idx") or "").strip().upper()
		if answer_idx in {"A", "B", "C", "D", "E"}:
			pred_token = _extract_mcq_token(pred)
			if pred_token:
				return 1 if pred_token == answer_idx else 0
		# Fallback: compare against answer text
		return 1 if _normalize_answer(pred) and _normalize_answer(pred) == _normalize_answer(gold_text) else 0
	if bench in {"gsm8k", "gsm"}:
		pnum = _extract_last_number(pred)
		gnum = _extract_last_number(gold_text)
		if pnum and gnum:
			return 1 if _canonical_num(pnum) == _canonical_num(gnum) else 0
	return 1 if _normalize_answer(pred) and _normalize_answer(pred) == _normalize_answer(gold_text) else 0


def _unwrap_agent_result(result: Any) -> str:
	if result is None:
		return ""
	if isinstance(result, (int, float, bool)):
		return ""
	if isinstance(result, str):
		return result.strip()
	if isinstance(result, dict):
		if "choices" in result:
			try:
				c0 = (result.get("choices") or [{}])[0]
				if isinstance(c0, dict):
					msg = c0.get("message") or {}
					if isinstance(msg, dict) and msg.get("content"):
						return str(msg["content"]).strip()
					if c0.get("text"):
						return str(c0["text"]).strip()
			except Exception:
				pass
		for k in ("final_result", "final_text", "text", "answer", "output", "content", "result"):
			if k in result and result[k]:
				return str(result[k]).strip()
		for v in result.values():
			if isinstance(v, str) and v.strip():
				return v.strip()
		return ""
	for attr in ("final_result", "final_text", "text", "answer", "output", "content"):
		if hasattr(result, attr):
			val = getattr(result, attr, None)
			if val:
				return str(val).strip()
	return str(result).strip()


def _extract_subtask_count(trace: Any) -> Tuple[int, Dict[str, Any]]:
	"""
	Extract per-task subtask count from trace.
	- Planner mode: use winning plan's chain length
	- Non-planner: count trace subtasks
	"""
	if not isinstance(trace, dict):
		return 0, {}

	# Planner mode: derive from plans/keys/weights/final
	plans = trace.get("plans", None)
	if isinstance(plans, list) and plans:
		keys = trace.get("keys", []) or []
		weights = trace.get("weights", []) or []
		win_key = trace.get("final", None)

		best_idx = None
		best_w = None
		if win_key is not None and keys and weights:
			for i, (k, w) in enumerate(zip(keys, weights)):
				if k == win_key:
					if best_w is None or float(w) > float(best_w):
						best_w = w
						best_idx = i

		if best_idx is None and weights:
			best_idx = max(range(len(weights)), key=lambda i: float(weights[i] or 0.0))
		if best_idx is None:
			best_idx = 0

		if 0 <= best_idx < len(plans):
			chain = plans[best_idx].get("chain", []) if isinstance(plans[best_idx], dict) else []
			count = len(chain) if isinstance(chain, list) else 0
			plan_lengths = [
				len(p.get("chain", []) or []) if isinstance(p, dict) else 0 for p in plans
			]
			return count, {"plan_index": best_idx, "plan_lengths": plan_lengths}

	# Non-planner: count trace subtasks
	traces = trace.get("traces", {}) or {}
	if isinstance(traces, dict):
		return len(traces), {}

	return 0, {}


def _resolve_openrouter_config_path(config_dir: str, aid: int) -> str:
	filename = f"config_agent_openrouter_{aid}.yaml"
	candidates = [
		os.path.join(config_dir, filename),
		os.path.join(config_dir, "configs", "openrouter", filename),
	]
	for p in candidates:
		if os.path.exists(p):
			return p
	# Search nested folders (e.g., runtime/configs/openrouter/<agent-name>/...)
	for root, _dirs, files in os.walk(config_dir):
		if filename in files:
			return os.path.join(root, filename)
	return candidates[0]


def _scan_openrouter_configs(config_dir: str) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
	"""
	Scan config_dir recursively and return:
	- id_to_path: {agent_id: file_path}
	- folder_to_ids: {folder_name: [agent_ids]}
	"""
	id_to_path: Dict[int, str] = {}
	folder_to_ids: Dict[str, List[int]] = {}
	if not os.path.isdir(config_dir):
		return id_to_path, folder_to_ids
	for root, _dirs, files in os.walk(config_dir):
		for name in files:
			if not (name.startswith("config_agent_openrouter_") and name.endswith(".yaml")):
				continue
			m = re.match(r"config_agent_openrouter_(\d+)\.yaml$", name)
			if not m:
				continue
			aid = int(m.group(1))
			path = os.path.join(root, name)
			id_to_path[aid] = path
			rel = os.path.relpath(root, config_dir)
			# Folder name is the first path segment under config_dir
			folder = rel.split(os.sep)[0] if rel and rel != "." else ""
			if folder:
				folder_to_ids.setdefault(folder, []).append(aid)
	for k in folder_to_ids:
		folder_to_ids[k] = sorted(set(folder_to_ids[k]))
	return id_to_path, folder_to_ids


def _parse_agent_ids(agent_arg: str, config_dir: str) -> List[int]:
	tokens = [t.strip() for t in (agent_arg or "").split(",") if t.strip()]
	if not tokens:
		return []
	id_to_path, folder_to_ids = _scan_openrouter_configs(config_dir)
	ids: List[int] = []
	seen = set()
	for t in tokens:
		# Support folder-qualified single agent, e.g. "deepseek-v3:16" or "deepseek-v3/16"
		m = re.match(r"^(.+?)[/:](\d+)$", t)
		if m:
			folder = m.group(1)
			aid = int(m.group(2))
			if folder in folder_to_ids and aid in folder_to_ids[folder]:
				if aid not in seen:
					ids.append(aid)
					seen.add(aid)
				continue
			raise ValueError(
				f"Agent {aid} not found under folder '{folder}' (runtime-dir: {config_dir})."
			)
		if t.isdigit():
			aid = int(t)
			if aid not in seen:
				ids.append(aid)
				seen.add(aid)
			continue
		if t in folder_to_ids:
			for aid in folder_to_ids[t]:
				if aid not in seen:
					ids.append(aid)
					seen.add(aid)
			continue
		raise ValueError(
			f"Unknown agent token '{t}'. Use numeric IDs or folder names under {config_dir}."
		)
	return ids


def load_agents_from_runtime(config_dir: str, agent_ids: List[int]) -> List[Agent]:
	agents: List[Agent] = []
	for aid in agent_ids:
		path = _resolve_openrouter_config_path(config_dir, aid)
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


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "a", encoding="utf-8") as f:
		f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_progress_state(path: str, record: Dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		f.write(json.dumps(record, ensure_ascii=False))


def _load_progress_state(outdir: str) -> Optional[Dict[str, Any]]:
	path = os.path.join(outdir, "progress_state.json")
	if not os.path.exists(path):
		return None
	with open(path, "r", encoding="utf-8") as f:
		return json.loads(f.read())


def run_phase(
	phase: str,
	tasks: List[Dict[str, Any]],
	start_index: int,
	cot_count: int,
	outdir: str,
	verbose: bool,
	print_each_step: bool,
	agents: Optional[List[Any]] = None,  # ✅ For cold_start round-robin
	requirements_override: Optional[List[str]] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
	logs: List[Dict[str, Any]] = []
	os.makedirs(outdir, exist_ok=True)
	progress_path = os.path.join(outdir, "progress.jsonl")
	progress_state_path = os.path.join(outdir, "progress_state.json")
	
	# ✅ Cold_start mode: round-robin agent assignment (each task -> one agent)
	# If phase is "cold_start" and agents provided, use round-robin
	use_cold_start_round_robin = (phase == "cold_start" and agents is not None)
	
	for i, task in enumerate(tasks, start=start_index):
		t0 = time.time()
		task_obj = build_task_obj(task, i=i, requirements=requirements_override)
		
		# ✅ Cold_start: inject task_index into context for round-robin selection
		if use_cold_start_round_robin:
			# Inject task_index (0-based from start_index) into context
			ctx = getattr(task_obj, "context", {}) or {}
			if not isinstance(ctx, dict):
				ctx = {}
			ctx["_cold_start_task_index"] = i - start_index  # 0-based index within this phase
			# Get agent keys using symphony's _resolve_agent_key (sorted for consistent order)
			agent_keys = []
			for ag in agents:
				aid = (
					str(getattr(ag, "agent_id", "")) or
					str(getattr(ag, "node_id", "")) or
					str(getattr(ag, "name", "")) or
					str(getattr(ag, "id", "")) or
					""
				).strip()
				if aid:
					agent_keys.append(aid)
			# ✅ Sort agent keys to ensure consistent round-robin order
			agent_keys.sort()
			ctx["_cold_start_agents"] = agent_keys  # Agent keys for round-robin (sorted)
			try:
				task_obj.context = ctx
			except Exception:
				pass
			# Force cot_count=1 for cold_start (each task runs once)
			actual_cot_count = 1
		else:
			actual_cot_count = cot_count

		# ✅ cold_start: do NOT decompose into subtasks; execute raw task once
		if phase == "cold_start":
			# pick agent by round-robin
			selected_agent = None
			selected_aid = "NA"
			if agents:
				agent_keys = []
				for ag in agents:
					aid = (
						str(getattr(ag, "agent_id", "")) or
						str(getattr(ag, "node_id", "")) or
						str(getattr(ag, "name", "")) or
						str(getattr(ag, "id", "")) or
						""
					).strip()
					if aid:
						agent_keys.append(aid)
				agent_keys.sort()
				if agent_keys:
					target_key = agent_keys[(i - start_index) % len(agent_keys)]
					for ag in agents:
						aid = (
							str(getattr(ag, "agent_id", "")) or
							str(getattr(ag, "node_id", "")) or
							str(getattr(ag, "name", "")) or
							str(getattr(ag, "id", "")) or
							""
						).strip()
						if aid == target_key:
							selected_agent = ag
							selected_aid = aid
							break
			if selected_agent is None and agents:
				selected_agent = agents[0]
				selected_aid = (
					str(getattr(selected_agent, "agent_id", "")) or
					str(getattr(selected_agent, "node_id", "")) or
					str(getattr(selected_agent, "name", "")) or
					str(getattr(selected_agent, "id", "")) or
					"NA"
				).strip()

			final_text = ""
			agent_ids = []
			subtask_count = 0
			subtask_meta: Dict[str, Any] = {"mode": "no_subtask"}

			if selected_agent is None:
				final_text = ""
			else:
				try:
					# Use legacy task dict to execute raw prompt
					reqs = list(getattr(task_obj, "requirements", []) or ["analysis"])
					req0 = str(reqs[0]) if reqs else "analysis"
					raw_prompt = task_to_text(task)
					legacy = {
						"subtask_id": 1,
						"steps": {"1": [raw_prompt, req0]},
						"previous_results": [],
						"original_problem": raw_prompt,
						"final_result": "",
						"user_id": "pretrain_cold_start",
					}
					result = selected_agent.execute_task(legacy)  # type: ignore[attr-defined]
					final_text = _unwrap_agent_result(result)
					agent_ids = [selected_aid]
				except Exception as e:
					final_text = f"[AGENT_ERROR] {str(e)}"

			dt = time.time() - t0
			ok = is_success(final_text)
			pred_raw = final_text
			pred = extract_pred(final_text)
			# Force medical_qa to use MCQ token if possible
			if str(task.get("benchmark", "")).strip().lower() == "medical_qa":
				mapped = _map_medical_qa_pred_to_token(task, pred)
				if mapped:
					pred = mapped
				# Previous behavior (kept for reference):
				# pred = extract_pred(final_text)
			if ok == 1 and not pred:
				pred = final_text.strip() if final_text else ""
			gold_text = extract_gold_text(task)
			acc = is_correct(pred, gold_text, str(task.get("benchmark", "")), task=task)
			logs.append(
				{
					"i": i,
					"phase": phase,
					"task_id": task.get("task_id") or task.get("id"),
					"benchmark": task.get("benchmark"),
					"difficulty_bin": task.get("difficulty_bin"),
					"agent_ids": agent_ids,
					"subtask_count": subtask_count,
					"subtask_meta": subtask_meta,
					"ok": ok,
					"pred_raw": pred_raw,
					"pred": pred,
					"gold": gold_text,
					"acc": acc,
					"latency_s": dt,
				}
			)
			if print_each_step:
				first_agent = agent_ids[0] if agent_ids else "NA"
				print(f"[{phase}] step={i} agent={first_agent} ok={ok} acc={acc} latency={dt:.2f}s")
			continue
		
		# ✅ Error handling: retry on transient errors (e.g., server_500)
		# Retry up to 3 times to avoid occasional upstream failures killing the entire experiment
		trace = None
		err_msg = None
		for attempt in range(3):
			try:
				trace = symphony_module.execute_task(task_obj, cot_count=actual_cot_count, return_mode="trace")
				break
			except Exception as e:
				err_msg = f"{type(e).__name__}: {str(e)}"
				# Simple backoff to reduce transient jitter
				if attempt < 2:  # Don't sleep on last attempt
					time.sleep(1.5 * (attempt + 1))
		
		# If still failed: record error log but don't crash
		if trace is None:
			dt = time.time() - t0
			gold_text = extract_gold_text(task)
			logs.append(
				{
					"i": i,
					"phase": phase,
					"task_id": task.get("task_id") or task.get("id"),
					"benchmark": task.get("benchmark"),
					"difficulty_bin": task.get("difficulty_bin"),
					"agent_ids": [],
					"ok": 0,
					"pred": "",
					"gold": gold_text,
					"acc": 0,
					"latency_s": dt,
					"error": err_msg,
				}
			)
			if print_each_step:
				print(f"[{phase}] step={i} agent=NA ok=0 acc=0 latency={dt:.2f}s err={err_msg}")
			continue
		
		final_text = ""
		agent_ids: List[str] = []
		subtask_count = 0
		subtask_meta: Dict[str, Any] = {}
		if isinstance(trace, dict):
			subtask_count, subtask_meta = _extract_subtask_count(trace)
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
		# ✅ Store both raw and clean pred: raw for debugging, clean for evaluation
		pred_raw = final_text  # Original voted output (may contain ```json fences)
		pred = extract_pred(final_text)  # Clean extracted answer
		# Force medical_qa to use MCQ token if possible
		if str(task.get("benchmark", "")).strip().lower() == "medical_qa":
			mapped = _map_medical_qa_pred_to_token(task, pred)
			if mapped:
				pred = mapped
			# Previous behavior (kept for reference):
			# pred = extract_pred(final_text)
		# ✅ Ensure pred is not empty if ok=1 (fallback to raw if extraction failed)
		if ok == 1 and not pred:
			# If extraction failed but text is valid, use stripped raw text as fallback
			pred = final_text.strip() if final_text else ""
		gold_text = extract_gold_text(task)
		acc = is_correct(pred, gold_text, str(task.get("benchmark", "")), task=task)
		logs.append(
			{
				"i": i,
				"phase": phase,
				"task_id": task.get("task_id") or task.get("id"),
				"benchmark": task.get("benchmark"),
				"difficulty_bin": task.get("difficulty_bin"),
				"agent_ids": agent_ids,
				"subtask_count": subtask_count,
				"subtask_meta": subtask_meta,
				"ok": ok,
				"pred_raw": pred_raw,  # ✅ Raw output for debugging
				"pred": pred,  # ✅ Clean answer for evaluation (used in acc calculation)
				"gold": gold_text,
				"acc": acc,
				"latency_s": dt,
			}
		)
		# ✅ Progress logging: append per-step + overwrite last state
		prog = {
			"phase": phase,
			"i": i,
			"task_id": task.get("task_id") or task.get("id"),
			"ok": ok,
			"acc": acc,
			"ts": datetime.datetime.now().isoformat(timespec="seconds"),
		}
		_append_jsonl(progress_path, prog)
		_write_progress_state(progress_state_path, prog)
		if print_each_step:
			first_agent = agent_ids[0] if agent_ids else "NA"
			print(f"[{phase}] step={i} agent={first_agent} ok={ok} acc={acc} latency={dt:.2f}s")
		elif verbose and (i % 10 == 0):
			print(f"[{phase}] {i} done, ok={ok}, acc={acc}, latency={dt:.2f}s")

	os.makedirs(outdir, exist_ok=True)
	log_path = os.path.join(outdir, f"pretrain_{phase}.jsonl")
	if os.path.exists(log_path):
		ts = datetime.datetime.now().strftime("%H%M%S")
		log_path = os.path.join(outdir, f"pretrain_{phase}_{ts}.jsonl")
	with open(log_path, "w", encoding="utf-8") as f:
		for r in logs:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")

	return start_index + len(tasks), logs


def _plot_acc_curve(logs: List[Dict[str, Any]], outdir: str, filename: str, title: str) -> None:
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
	plt.title(title)
	plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
	plt.legend(frameon=False, fontsize=9)
	os.makedirs(outdir, exist_ok=True)
	acc_path = os.path.join(outdir, filename)
	if os.path.exists(acc_path):
		ts = datetime.datetime.now().strftime("%H%M%S")
		base, ext = os.path.splitext(filename)
		acc_path = os.path.join(outdir, f"{base}_{ts}{ext or '.png'}")
	plt.savefig(acc_path)
	plt.close()


def plot_acc_curves_by_phase(all_logs: List[Dict[str, Any]], outdir: str) -> None:
	if not all_logs:
		return
	train_logs = [r for r in all_logs if str(r.get("phase")) in {"cold_start", "pretrain"}]
	test_logs = [r for r in all_logs if str(r.get("phase")) == "test"]
	if train_logs:
		_plot_acc_curve(train_logs, outdir, "acc_curve_train.png", "ACC Curve (cold_start + pretrain)")
	if test_logs:
		_plot_acc_curve(test_logs, outdir, "acc_curve_test.png", "ACC Curve (test)")


def write_accuracy_summary(all_logs: List[Dict[str, Any]], outdir: str) -> None:
	if not all_logs:
		return

	# Compute cumulative ACC per phase (avoid mixing cold_start/pretrain/test)
	sorted_logs = sorted(all_logs, key=lambda x: int(x.get("i", 0)))
	cum_map: Dict[Tuple[str, str], float] = {}
	phase_counts: Dict[str, int] = {}
	phase_totals: Dict[str, int] = {}
	for r in sorted_logs:
		phase = str(r.get("phase") or "unknown")
		task_id = str(r.get("task_id") or "")
		phase_counts[phase] = phase_counts.get(phase, 0) + 1
		phase_totals[phase] = phase_totals.get(phase, 0) + int(r.get("acc", r.get("ok", 0)))
		cum_map[(phase, task_id)] = phase_totals[phase] / max(1, phase_counts[phase])

	rows: List[Dict[str, Any]] = []
	for r in all_logs:
		agent_ids = r.get("agent_ids") or []
		if isinstance(agent_ids, list):
			node_id = ",".join([str(x) for x in agent_ids if x])
		else:
			node_id = str(agent_ids) if agent_ids else ""
		phase = str(r.get("phase") or "unknown")
		task_id = str(r.get("task_id") or "")
		rows.append(
			{
				"phase": phase,
				"task_id": task_id,
				"node_id": node_id,
				"subtask_count": int(r.get("subtask_count") or 0),
				"acc": int(r.get("acc") or 0),
				"ok": int(r.get("ok") or 0),
				"cum_acc": cum_map.get((phase, task_id), 0.0),
			}
		)

	os.makedirs(outdir, exist_ok=True)
	csv_path = os.path.join(outdir, "accuracy_summary.csv")
	with open(csv_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"phase",
				"task_id",
				"node_id",
				"subtask_count",
				"acc",
				"ok",
				"cum_acc",
			],
		)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--task-pool", type=str, required=True, help="Path to JSONL task pool")
	ap.add_argument("--benchmark", type=str, default=None, help="Filter tasks by benchmark (e.g., bbh)")
	ap.add_argument("--n", type=int, default=None, help="Total tasks to use")
	ap.add_argument("--seed", type=int, default=123)
	ap.add_argument("--cold-n", type=int, default=50)
	ap.add_argument("--pretrain-n", type=int, default=200)
	ap.add_argument("--test-n", type=int, default=250, help="Number of test tasks (no validation stage)")
	ap.add_argument("--val-n", type=int, default=None, help="Deprecated: use --test-n instead")
	ap.add_argument("--topL", type=int, default=3)
	ap.add_argument("--ucb-alpha", type=float, default=1.0)
	ap.add_argument("--ucb-l2", type=float, default=1.0)
	ap.add_argument("--cot-count", type=int, default=1)
	ap.add_argument("--plan-k", type=int, default=1)
	ap.add_argument("--planner-decompose", action="store_true",
	                help="Use planner to decompose tasks even when plan_k=1 (prompt-based subtasks)")
	ap.add_argument("--multi-plan", action="store_true", help="Use all agents as planners (plan_k = agent count)")
	ap.add_argument("--agents", type=str, default="11,12,13,14,15",
	                help="Comma-separated agent IDs or folder names under runtime-dir (default: 11,12,13,14,15)")
	ap.add_argument("--requirements", type=str, default="",
	                help="Comma-separated requirements override (default: use task field or ['analysis'])")
	ap.add_argument("--outdir", type=str, default="pretrain_results")
	ap.add_argument("--resume-dir", type=str, default=None, help="Resume from existing outdir (uses progress_state.json)")
	ap.add_argument("--runtime-dir", type=str, default="runtime")
	ap.add_argument("--save-selector", type=str, default=None, help="Save UCB state after pretrain")
	ap.add_argument("--load-selector", type=str, default=None, help="Load UCB state and run test only")
	ap.add_argument("--plot-acc", action="store_true", help="Plot cumulative ACC curve")
	ap.add_argument("--print-each-step", action="store_true", help="Print agent per step")
	ap.add_argument("--verbose", action="store_true")
	args = ap.parse_args()
	# 默认每步打印，便于观察 agent 是否在响应
	if not args.print_each_step:
		args.print_each_step = True

	if args.resume_dir:
		args.outdir = args.resume_dir
	else:
		date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		args.outdir = os.path.join(args.outdir, date_str)

	tasks = load_tasks(args.task_pool, args.n, seed=args.seed, benchmark=args.benchmark)

	if args.val_n is not None:
		print("[WARN] --val-n is deprecated. Use --test-n instead.")
		args.test_n = int(args.val_n)

	if args.load_selector:
		args.cold_n = 0
		args.pretrain_n = 0

	total_needed = args.cold_n + args.pretrain_n + args.test_n
	if total_needed > len(tasks):
		args.test_n = max(0, len(tasks) - args.cold_n - args.pretrain_n)

	cold_tasks = tasks[: args.cold_n]
	pretrain_tasks = tasks[args.cold_n : args.cold_n + args.pretrain_n]
	test_tasks = tasks[args.cold_n + args.pretrain_n : args.cold_n + args.pretrain_n + args.test_n]

	# load agents from command-line argument
	agent_ids = _parse_agent_ids(args.agents, args.runtime_dir)
	agents = load_agents_from_runtime(args.runtime_dir, agent_ids)
	for ag in agents:
		symphony_module.register_agent(ag)
	if args.multi_plan:
		args.plan_k = max(1, len(agents))

	idx = 1
	if args.resume_dir:
		state = _load_progress_state(args.outdir)
		if not state or "i" not in state:
			raise RuntimeError(f"resume-dir set but progress_state.json missing or invalid: {args.outdir}")
		last_i = int(state.get("i") or 0)
		if last_i > 0:
			cold_done = min(args.cold_n, max(0, last_i))
			pretrain_done = min(args.pretrain_n, max(0, last_i - args.cold_n))
			test_done = min(args.test_n, max(0, last_i - args.cold_n - args.pretrain_n))
			cold_tasks = cold_tasks[cold_done:]
			pretrain_tasks = pretrain_tasks[pretrain_done:]
			test_tasks = test_tasks[test_done:]
			idx = last_i + 1
	all_logs: List[Dict[str, Any]] = []
	req_override: Optional[List[str]] = None
	if args.requirements:
		req_override = [r.strip() for r in args.requirements.split(",") if r.strip()]

	if args.load_selector:
		selector = load_selector(args.load_selector)
		symphony_module._global_orchestrator.selector = FrozenSelector(selector)
		symphony_module._global_orchestrator.use_dynamic = True
		if test_tasks:
			idx, logs = run_phase("test", test_tasks, idx, args.cot_count, args.outdir, args.verbose, args.print_each_step, requirements_override=req_override)
			all_logs.extend(logs)
	else:
		# cold start: static Top-L (no planner, no multi-CoT)
		symphony_module.init(
			use_dynamic=False,
			topL=int(args.topL),
			# plan_k=int(args.plan_k),
			plan_k=1,  # ✅ B: cold_start 不启用 planner
		)
		if cold_tasks:
			# idx, logs = run_phase("cold_start", cold_tasks, idx, args.cot_count, args.outdir, args.verbose, args.print_each_step, agents=agents) 
			idx, logs = run_phase("cold_start", cold_tasks, idx, 1, args.outdir, args.verbose, args.print_each_step, agents=agents, requirements_override=req_override)  # ✅ C: cold_start 强制 cot_count=1
			all_logs.extend(logs)

		# pretrain: Top-L + UCB (updates enabled)
		symphony_module.init(
			use_dynamic=True,
			topL=int(args.topL),
			linucb_alpha=float(args.ucb_alpha),
			linucb_l2=float(args.ucb_l2),
			plan_k=int(args.plan_k),
			use_planner_decompose=bool(args.planner_decompose),
		)
		if pretrain_tasks:
			idx, logs = run_phase("pretrain", pretrain_tasks, idx, args.cot_count, args.outdir, args.verbose, args.print_each_step, requirements_override=req_override)
			all_logs.extend(logs)

		selector = symphony_module._global_orchestrator.selector
		if selector is None:
			raise RuntimeError("UCB selector is not initialized; cannot freeze for test")
		if args.save_selector:
			save_path = args.save_selector
			if not os.path.isabs(save_path):
				save_path = os.path.join(args.outdir, save_path)
			if os.path.exists(save_path):
				base, ext = os.path.splitext(save_path)
				ts = datetime.datetime.now().strftime("%H%M%S")
				save_path = f"{base}_{ts}{ext or '.json'}"
			save_selector(selector, save_path)

		symphony_module._global_orchestrator.selector = FrozenSelector(selector)
		symphony_module._global_orchestrator.use_dynamic = True

		if test_tasks:
			idx, logs = run_phase("test", test_tasks, idx, args.cot_count, args.outdir, args.verbose, args.print_each_step, requirements_override=req_override)
			all_logs.extend(logs)

	if args.plot_acc:
		plot_acc_curves_by_phase(all_logs, args.outdir)
	write_accuracy_summary(all_logs, args.outdir)

	print(f"[OK] Pre-train done. outdir={args.outdir}")


if __name__ == "__main__":
	main()
