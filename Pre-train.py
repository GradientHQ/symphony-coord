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

def _stratified_sample_bbh(tasks: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    # Sample evenly across BBH task types (task_name) for stability
    if n <= 0:
        return tasks
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        meta = t.get("scorer_metadata") or {}
        name = str(meta.get("task_name") or "").strip() or "unknown"
        groups.setdefault(name, []).append(t)
    if not groups:
        return tasks

    rng = random.Random(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    base = n // len(group_keys)
    remainder = n % len(group_keys)

    alloc: Dict[str, int] = {k: base for k in group_keys}
    for k in group_keys[:remainder]:
        alloc[k] += 1

    selected: List[Dict[str, Any]] = []
    deficit = 0
    capacity: Dict[str, int] = {}
    for k in group_keys:
        pool = groups[k]
        rng.shuffle(pool)
        take = min(len(pool), alloc[k])
        selected.extend(pool[:take])
        if take < alloc[k]:
            deficit += (alloc[k] - take)
        capacity[k] = max(0, len(pool) - take)

    if deficit > 0:
        fill_keys = [k for k in group_keys if capacity.get(k, 0) > 0]
        rng.shuffle(fill_keys)
        idx = 0
        while deficit > 0 and fill_keys:
            k = fill_keys[idx % len(fill_keys)]
            if capacity[k] <= 0:
                idx += 1
                continue
            pool = groups[k]
            start = len(pool) - capacity[k]
            selected.append(pool[start])
            capacity[k] -= 1
            deficit -= 1
            idx += 1

    return selected

def _stratified_split_bbh_by_phase(
    tasks: List[Dict[str, Any]],
    cold_n: int,
    pretrain_n: int,
    test_n: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        meta = t.get("scorer_metadata") or {}
        name = str(meta.get("task_name") or "").strip() or "unknown"
        groups.setdefault(name, []).append(t)
    if not groups:
        # Fallback to sequential split
        cold = tasks[:cold_n]
        pre = tasks[cold_n:cold_n + pretrain_n]
        test = tasks[cold_n + pretrain_n:cold_n + pretrain_n + test_n]
        return cold, pre, test

    rng = random.Random(seed)
    group_keys = sorted(groups.keys())
    for k in group_keys:
        rng.shuffle(groups[k])

    def _alloc(phase_n: int, salt: int) -> Dict[str, int]:
        if phase_n <= 0:
            return {k: 0 for k in group_keys}
        keys = list(group_keys)
        random.Random(seed + salt).shuffle(keys)
        base = phase_n // len(keys)
        remainder = phase_n % len(keys)
        alloc = {k: base for k in keys}
        for k in keys[:remainder]:
            alloc[k] += 1
        return alloc

    alloc_cold = _alloc(cold_n, 101)
    alloc_pre = _alloc(pretrain_n, 202)
    alloc_test = _alloc(test_n, 303)

    idx_map = {k: 0 for k in group_keys}

    def _take(alloc: Dict[str, int], salt: int) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        deficit = 0
        capacity: Dict[str, int] = {}
        for k in group_keys:
            pool = groups[k]
            start = idx_map[k]
            take = min(max(0, len(pool) - start), alloc.get(k, 0))
            if take:
                selected.extend(pool[start:start + take])
            idx_map[k] = start + take
            if take < alloc.get(k, 0):
                deficit += (alloc.get(k, 0) - take)
            capacity[k] = max(0, len(pool) - idx_map[k])

        if deficit > 0:
            fill_keys = [k for k in group_keys if capacity.get(k, 0) > 0]
            random.Random(seed + salt).shuffle(fill_keys)
            idx = 0
            while deficit > 0 and fill_keys:
                k = fill_keys[idx % len(fill_keys)]
                if capacity[k] <= 0:
                    idx += 1
                    continue
                pool = groups[k]
                selected.append(pool[idx_map[k]])
                idx_map[k] += 1
                capacity[k] -= 1
                deficit -= 1
                idx += 1
        return selected

    cold = _take(alloc_cold, 404)
    pre = _take(alloc_pre, 505)
    test = _take(alloc_test, 606)
    return cold, pre, test


def _normalize_bbh_task_types(raw_types: Optional[List[str]]) -> List[str]:
    if not raw_types:
        return []
    cleaned = []
    for t in raw_types:
        name = str(t or "").strip()
        if not name:
            continue
        cleaned.append(name.lower())
    return sorted(set(cleaned))

def _filter_bbh_tasks_by_type(
    tasks: List[Dict[str, Any]],
    bbh_task_types: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    ✅ Fix 7: When bbh_task_types is set, discard non-BBH tasks to avoid mixing benchmarks.
    """
    allowed = _normalize_bbh_task_types(bbh_task_types)
    if not allowed:
        return tasks
    filtered: List[Dict[str, Any]] = []
    for t in tasks:
        is_bbh = str(t.get("benchmark") or "").strip().lower() == "bbh"
        # If bbh_task_types is specified, only keep BBH tasks (discard non-BBH)
        if bbh_task_types and not is_bbh:
            continue  # Discard non-BBH tasks when filtering by BBH task types
        if not is_bbh:
            # If no bbh_task_types filter, keep all tasks (backward compatibility)
            if not bbh_task_types:
                filtered.append(t)
            continue
        meta = t.get("scorer_metadata") or {}
        task_name = str(meta.get("task_name") or "").strip().lower()
        if task_name in allowed:
            filtered.append(t)
    return filtered

def load_tasks(
    path: str,
    n: Optional[int],
    seed: int,
    benchmark: Optional[str],
    bbh_task_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if benchmark:
                obj_bench = str(obj.get("benchmark") or "").strip().lower()
                bench = str(benchmark).strip().lower()
                if obj_bench != bench:
                    continue
            tasks.append(obj)

    if bbh_task_types:
        tasks = _filter_bbh_tasks_by_type(tasks, bbh_task_types)

    if n is None or n <= 0 or n >= len(tasks):
        return tasks

    if (benchmark or "").strip().lower() == "bbh":
        return _stratified_sample_bbh(tasks, int(n), seed)

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
    """
    ✅ Fix 5: Consistent normalization of requirements.
    - Convert to lowercase
    - Replace '-' with '_' for consistency
    - Remove empty strings
    - Deduplicate while preserving order
    """
    if not reqs:
        return ["analysis"]
    # Normalize: lowercase, replace '-' with '_', filter empty
    normalized = []
    seen = set()
    for r in reqs:
        if not r:
            continue
        # Normalize: lowercase and replace '-' with '_'
        norm = str(r).strip().lower().replace("-", "_")
        if norm and norm not in seen:
            normalized.append(norm)
            seen.add(norm)
    return normalized if normalized else ["analysis"]

def _bbh_answer_format(task: Dict[str, Any]) -> str:
    meta = task.get("scorer_metadata") or {}
    tname = str(meta.get("task_name") or "").strip()
    if tname in {"boolean_expressions"}:
        return "Return ONLY True or False."
    if tname in {"causal_judgement", "navigate", "web_of_lies"}:
        return "Return ONLY Yes or No."
    if tname in {"sports_understanding"}:
        return "Return ONLY yes or no."
    if tname in {
        "date_understanding",
        "disambiguation_qa",
        "logical_deduction",
        "tracking_shuffled_objects",
        "movie_recommendation",
    }:
        return "Return ONLY the option label in parentheses, e.g., (A)."
    return ""

def build_task_obj(
    task: Dict[str, Any],
    i: int,
    requirements: Optional[List[str]] = None,
    solution_mode: Optional[str] = None,
) -> Task:
    task_text = task_to_text(task)
    if solution_mode:
        mode = str(solution_mode).strip()
        if mode:
            task_text = f"SOLUTION_MODE: {mode}\n\n" + task_text
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
            + "Return ONLY the single token (e.g., A). No extra words. If unsure, still output one token."
        )
        # Previous behavior (kept for reference):
        # task_text = task_text  # free-form answer text (may cause acc=0 for medical_qa)
    if bench in {"gsm8k", "gsm"}:
        # 针对nano模型的GSM8K专用提示词：短步骤+强制自检+只输出整数
        # 添加few-shot examples（2-4个短例子）
        # ✅ Fix 6: Remove "Solution:" prefix from few-shot examples to avoid teaching models to output extra text
        few_shot_examples = """[FEW-SHOT EXAMPLES]

Example 1:
Problem: Janet has 3 apples. She gives away 2 apples. How many apples does she have left?
{"final_answer": "1", "valid": 1, "confidence": 1.0}

Example 2:
Problem: Tom has 5 books. He buys 3 more books. How many books does he have now?
{"final_answer": "8", "valid": 1, "confidence": 1.0}

Example 3:
Problem: A box contains 12 cookies. If 4 cookies are eaten, how many cookies remain?
{"final_answer": "8", "valid": 1, "confidence": 1.0}

Example 4:
Problem: Sarah has $20. She spends $7 on lunch. How much money does she have left?
{"final_answer": "13", "valid": 1, "confidence": 1.0}

Note: The examples above show the expected output format. You should think through the problem internally, but output ONLY the JSON object with no "Solution:" or "Output:" prefixes.
"""
        
        task_text = (
            task_text
            + "\n\n" + few_shot_examples
            + "\n[SOLVING_PROTOCOL FOR GSM8K]\n"
            + "CRITICAL: Follow this EXACT protocol. Do NOT deviate.\n\n"
            + "STEP 1: Read the problem. Identify: what number is being asked?\n"
            + "STEP 2: Break into 2-3 short steps MAX. Write each step as: [number] [operation] [number] = [result]\n"
            + "STEP 3: Quick self-check: plug your key intermediate numbers back into the problem. Do they make sense?\n"
            + "STEP 4: Extract the FINAL INTEGER answer only (no decimals, no fractions, no units).\n"
            + "STEP 5: Output ONLY one JSON object. No extra text.\n\n"
            + "[OUTPUT FORMAT: STRICT JSON ONLY]\n"
            + "You MUST output EXACTLY one JSON object with this structure:\n"
            + '{"final_answer": "<integer_as_string>", "valid": 1, "confidence": <0.0-1.0>}\n\n'
            + "CRITICAL RULES:\n"
            + "1. final_answer MUST be an integer (whole number). Convert fractions/decimals to integers if needed.\n"
            + "2. Output ONLY the JSON object. NO text before, NO text after, NO multiple JSON objects.\n"
            + "3. If you see multiple JSON objects or extra text, your output is INVALID.\n"
            + "4. valid=1 only if format is correct AND answer is an integer.\n"
            + "5. Before outputting, do a quick sanity check: does your answer make sense in the problem context?\n\n"
            + "[EXAMPLE OUTPUT]\n"
            + '{"final_answer": "42", "valid": 1, "confidence": 0.9}\n'
        )
    if bench == "bbh":
        # Force per-task-type answer format to reduce parsing errors
        fmt = _bbh_answer_format(task)
        if fmt:
            task_text = task_text + "\n\n[ANSWER_FORMAT]\n" + fmt
    # 改动4: GSM8K应该走math_reasoning，不要混进general_reasoning
    task_reqs = requirements
    if task_reqs is None:
        # Prefer task-provided requirements if present
        task_reqs = task.get("requirements") if isinstance(task, dict) else None
    
    # 对于GSM8K，强制使用math_reasoning
    if bench in {"gsm8k", "gsm"}:
        if task_reqs:
            # 将general_reasoning替换为math_reasoning
            task_reqs = [r if r.lower() not in {"general_reasoning", "general-reasoning", "analysis"} 
                        else "math_reasoning" for r in task_reqs]
            if "math_reasoning" not in [r.lower() for r in task_reqs]:
                task_reqs = ["math_reasoning"] + task_reqs
        else:
            task_reqs = ["math_reasoning"]
    
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

def _bbh_task_name(task: Optional[Dict[str, Any]]) -> str:
    if not task:
        return ""
    meta = task.get("scorer_metadata") or {}
    return str(meta.get("task_name") or "").strip().lower()

def _bbh_extract_choice_token(text: str) -> str:
    if not text:
        return ""
    import re
    s = text.strip().upper()
    if not s:
        return ""
    m = re.search(r"^\(?([A-F])\)?", s)
    if m:
        return m.group(1)
    m = re.search(r"\(([A-F])\)", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-F])\b", s)
    if m:
        return m.group(1)
    return ""

def _bbh_normalize_yesno(text: str) -> str:
    if not text:
        return ""
    import re
    s = text.strip().lower()
    if not s:
        return ""
    m = re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else ""

def _bbh_normalize_truefalse(text: str) -> str:
    if not text:
        return ""
    import re
    s = text.strip().lower()
    if not s:
        return ""
    m = re.search(r"\b(true|false)\b", s)
    return m.group(1) if m else ""

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
    # normalize unicode minus/dash to ASCII minus
    clean = clean.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", clean)
    return _canonical_num(nums[-1]) if nums else ""

def _extract_gsm8k_final(text: str) -> str:
    """
    严格提取 GSM8K 的最终答案。
    
    优先级：
    1. 优先匹配 `#### <number>` 格式（GSM8K 标准格式）
    2. 清理中间计算块（如 `<<...>>`）避免抓取中间数字
    3. Fallback：取末尾附近的最后一个独立整数
    """
    if not text:
        return ""
    import re
    
    # 步骤1：清理中间计算块（如 <<56+40=96>>96 这种结构）
    # 移除所有 <<...>> 块，避免抓取中间计算数字
    text_clean = re.sub(r'<<[^>]*>>', '', text)
    
    # 步骤2：优先匹配 `#### <number>` 格式（GSM8K 标准格式）
    # 匹配模式：#### 后面跟着可选的空白，然后是数字（可能带负号、小数点）
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text_clean)
    if match:
        num_str = match.group(1).strip()
        # 对于 GSM8K，最终答案应该是整数，但允许小数（转换为整数）
        try:
            num_val = float(num_str)
            # 如果是整数，返回整数字符串；否则返回规范化的小数
            if abs(num_val - round(num_val)) < 1e-9:
                return str(int(round(num_val)))
            return _canonical_num(str(num_val))
        except (ValueError, TypeError, OverflowError):
            pass
    
    # 步骤3：如果没有 `####`，尝试在末尾附近找最后一个独立数字
    # 取最后 200 个字符（避免处理过长的文本）
    tail = text_clean[-200:] if len(text_clean) > 200 else text_clean
    
    # 移除所有 <<...>> 块（再次清理，确保 tail 中没有）
    tail = re.sub(r'<<[^>]*>>', '', tail)
    
    # 查找最后一个独立的整数（前后不是数字/小数点）
    # 匹配模式：负号可选，然后是整数（不允许小数点，因为 GSM8K 答案通常是整数）
    matches = list(re.finditer(r'(?<![0-9.-])(-?\d+)(?![0-9.])', tail))
    if matches:
        # 取最后一个匹配的整数
        last_match = matches[-1]
        num_str = last_match.group(1).strip()
        try:
            num_val = int(num_str)
            return str(num_val)
        except (ValueError, TypeError, OverflowError):
            pass
    
    # 步骤4：如果还是没有找到，尝试分数（如 1/2）
    frac = re.findall(r'(-?\d+)\s*/\s*(-?\d+)', tail)
    if frac:
        num, den = frac[-1]
        try:
            result = float(num) / float(den)
            if abs(result - round(result)) < 1e-9:
                return str(int(round(result)))
            return _canonical_num(str(result))
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            pass
    
    # Fallback：使用原来的 _extract_last_number（但已经清理了中间计算块）
    return _extract_last_number(tail)

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

    # 2) try parse as JSON (allow trailing garbage like concatenated JSON)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            for key in ["final_answer", "answer", "final", "output"]:
                if key in obj and obj[key] is not None:
                    return str(obj[key]).strip()
    except Exception:
        pass

    # 3) tolerate concatenated JSON: parse first object only
    try:
        decoder = json.JSONDecoder()
        start = t.find("{")
        if start != -1:
            obj, _ = decoder.raw_decode(t[start:])
            if isinstance(obj, dict):
                for key in ["final_answer", "answer", "final", "output"]:
                    if key in obj and obj[key] is not None:
                        return str(obj[key]).strip()
    except Exception:
        pass

    # 4) regex fallback for malformed JSON blobs
    try:
        m = re.search(r'final_answer"\s*:\s*"([^"]+)"', t)
        if m:
            return m.group(1).strip()
        m = re.search(r"final_answer'\s*:\s*'([^']+)'", t)
        if m:
            return m.group(1).strip()
        m = re.search(r"final_answer\s*:\s*(-?\d+(?:\.\d+)?)", t)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ""

def extract_pred(text: str) -> str:
    if not text:
        return ""
    json_ans = _try_json_final_answer(text)
    if json_ans:
        return json_ans
    # Handle concatenated JSON like {...}{...}
    if "}{" in text:
        json_ans = _try_json_final_answer(text.replace("}{", "} {"))
        if json_ans:
            return json_ans
    low = text.lower()
    key = "final answer:"
    if key in low:
        idx = low.rfind(key)
        tail = text[idx + len(key):].strip()
        json_tail = _try_json_final_answer(tail)
        return json_tail or tail
    if "####" in text:
        tail = text.split("####")[-1].strip()
        json_tail = _try_json_final_answer(tail)
        return json_tail or tail
    tail = text.strip()
    json_tail = _try_json_final_answer(tail)
    return json_tail or tail

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

def _map_medical_qa_gold_to_token(task: Optional[Dict[str, Any]], gold_text: str) -> str:
    if not task:
        return ""
    raw = task.get("raw_data") or {}
    answer_idx = str(raw.get("answer_idx") or "").strip().upper()
    if answer_idx in {"A", "B", "C", "D", "E"}:
        return answer_idx
    # Fallback: map gold text to option token if possible
    if not gold_text:
        return ""
    opts = raw.get("options") or {}
    if not isinstance(opts, dict) or not opts:
        return ""
    norm_gold = _normalize_answer(gold_text)
    if not norm_gold:
        return ""
    for k, v in opts.items():
        key = str(k).strip().upper()
        if key not in {"A", "B", "C", "D", "E"}:
            continue
        norm_opt = _normalize_answer(v)
        if norm_opt and (norm_opt == norm_gold or norm_gold in norm_opt or norm_opt in norm_gold):
            return key
    return ""


def is_correct(pred: str, gold_text: str, benchmark: str, task: Optional[Dict[str, Any]] = None) -> int:
    bench = (benchmark or "").strip().lower()
    if bench in {"humaneval", "human_eval"}:
        return _eval_humaneval(pred, task)
    if bench in {"bbh"}:
        tname = _bbh_task_name(task)
        if tname in {"causal_judgement", "navigate", "web_of_lies", "sports_understanding"}:
            p = _bbh_normalize_yesno(pred)
            g = _bbh_normalize_yesno(gold_text)
            return 1 if p and g and p == g else 0
        if tname in {"boolean_expressions"}:
            p = _bbh_normalize_truefalse(pred)
            g = _bbh_normalize_truefalse(gold_text)
            return 1 if p and g and p == g else 0
        if tname in {
            "date_understanding",
            "disambiguation_qa",
            "logical_deduction",
            "tracking_shuffled_objects",
            "movie_recommendation",
        }:
            p = _bbh_extract_choice_token(pred)
            g = _bbh_extract_choice_token(gold_text)
            return 1 if p and g and p == g else 0
    if bench in {"medical_qa"}:
        pred_token = _extract_mcq_token(pred) or _map_medical_qa_pred_to_token(task, pred)
        gold_token = _map_medical_qa_gold_to_token(task, gold_text)
        if pred_token and gold_token:
            return 1 if pred_token == gold_token else 0
        # Lenient fallback: compare against answer text
        return 1 if _normalize_answer(pred) and _normalize_answer(pred) == _normalize_answer(gold_text) else 0
    if bench in {"gsm8k", "gsm"}:
        pnum = _extract_gsm8k_final(pred)
        gnum = _extract_gsm8k_final(gold_text)
        if pnum and gnum:
            # 改动6: 轻量校验器 - 基本sanity check
            try:
                p_val = float(pnum.replace(",", "").replace("$", "").replace("%", ""))
                # 基本合理性检查：答案不应为负数（除非题目明确涉及负数）
                # 这里只做最基本的检查，更复杂的校验可以在调用前进行
                if p_val < 0:
                    # 对于大多数GSM8K问题，答案应该是非负的
                    # 但为了不误判，这里只记录，不直接判错
                    pass
            except (ValueError, TypeError):
                pass
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
                    # Fallback: reasoning-only responses
                    if isinstance(msg, dict):
                        reasoning = msg.get("reasoning")
                        if isinstance(reasoning, str) and reasoning.strip():
                            return reasoning.strip()
                        if isinstance(reasoning, dict):
                            summary = reasoning.get("summary")
                            if summary:
                                return str(summary).strip()
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

def _extract_final_text_from_trace(trace: Any) -> Tuple[str, List[str]]:
    if not isinstance(trace, dict):
        return "", []

    agent_ids: List[str] = []

    for k in ("final_result", "final_text", "answer", "output", "content"):
        v = trace.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip(), agent_ids

    plans = trace.get("plans")
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

        if 0 <= best_idx < len(plans) and isinstance(plans[best_idx], dict):
            p = plans[best_idx]
            p_trace = p.get("trace", {})
            if isinstance(p_trace, dict):
                for k in ("final_result", "final_text", "answer", "output", "content"):
                    v = p_trace.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip(), agent_ids
                steps = p_trace.get("steps", [])
                if isinstance(steps, list) and steps:
                    last = steps[-1]
                    if isinstance(last, dict):
                        for k in ("text", "output", "content", "final_text", "final_result", "answer", "result"):
                            v = last.get(k)
                            if isinstance(v, str) and v.strip():
                                return v.strip(), agent_ids

    traces = trace.get("traces", {}) or {}
    if isinstance(traces, dict) and traces:
        for _sid, st in traces.items():
            if not isinstance(st, dict):
                continue
            voted = st.get("voted") or st.get("voted_final")
            if isinstance(voted, str) and voted.strip():
                return voted.strip(), agent_ids

            runs = st.get("runs", []) or []
            if isinstance(runs, list):
                for r in runs:
                    if isinstance(r, dict):
                        aid = r.get("agent_id")
                        if aid:
                            agent_ids.append(str(aid))
                        for k in ("text", "output", "content", "final_text", "final_result", "answer", "result"):
                            v = r.get(k)
                            if isinstance(v, str) and v.strip():
                                return v.strip(), agent_ids

    return "", agent_ids

def _print_trace_details(phase: str, step_i: int, trace: Any) -> None:
    if not isinstance(trace, dict):
        return
    plans = trace.get("plans")
    if isinstance(plans, list) and plans:
        for p_idx, p in enumerate(plans, 1):
            if not isinstance(p, dict):
                continue
            planner = p.get("planner", "")
            w = p.get("w", "")
            chain = p.get("chain", [])
            subtask_count = len(chain) if isinstance(chain, list) else 0
            print(f"[{phase}] step={step_i} plan={p_idx} planner={planner} w={w} subtask_count={subtask_count}")
            if isinstance(chain, list):
                for s_idx, st in enumerate(chain, 1):
                    if not isinstance(st, dict):
                        continue
                    sid = st.get("id", "")
                    req = str(st.get("requirement", "")).replace("-", "_")
                    print(f"[{phase}] step={step_i} plan={p_idx} subtask={s_idx} id={sid} req={req}")
            tr = p.get("trace", {})
            steps = tr.get("steps", []) if isinstance(tr, dict) else []
            if isinstance(steps, list):
                for s_idx, st in enumerate(steps, 1):
                    if not isinstance(st, dict):
                        continue
                    subtask_id = st.get("subtask_id", "")
                    meta = st.get("meta", {}) if isinstance(st.get("meta", {}), dict) else {}
                    selected = meta.get("selected", "")
                    match_score = meta.get("match_score", "")
                    print(
                        f"[{phase}] step={step_i} plan={p_idx} step={s_idx} "
                        f"subtask_id={subtask_id} selected={selected} match_score={match_score}"
                    )

    traces = trace.get("traces", {})
    if isinstance(traces, dict):
        for sid, st in traces.items():
            if not isinstance(st, dict):
                continue
            runs = st.get("runs", [])
            if not isinstance(runs, list):
                continue
            for r_idx, r in enumerate(runs, 1):
                if not isinstance(r, dict):
                    continue
                aid = r.get("agent_id", "")
                match_score = r.get("match_score", "")
                latency_ms = r.get("latency_ms", "")
                print(
                    f"[{phase}] step={step_i} subtask={sid} run={r_idx} "
                    f"agent={aid} match_score={match_score} latency_ms={latency_ms}"
                )
                if "text" in r:
                    print(f"    output: {r.get('text')}")

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

def _agent_tag_from_arg(agent_arg: str, config_dir: str) -> str:
    tokens = [t.strip() for t in (agent_arg or "").split(",") if t.strip()]
    if not tokens:
        return "agents"
    _id_to_path, folder_to_ids = _scan_openrouter_configs(config_dir)
    parts: List[str] = []
    for t in tokens:
        # folder-qualified token: deepseek-v3:16 or deepseek-v3/16
        m = re.match(r"^(.+?)[/:](\d+)$", t)
        if m:
            parts.append(m.group(1))
            continue
        if t.isdigit():
            parts.append(f"id{t}")
            continue
        if t in folder_to_ids:
            parts.append(t)
            continue
        parts.append(t)
    # de-dup while preserving order
    seen: set[str] = set()
    uniq = []
    for p in parts:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    safe = "+".join(uniq).replace("/", "_")
    return safe or "agents"

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
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
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

def _to_list(v: Any) -> Any:
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return [ _to_list(x) for x in v ]
    return v

def _append_ucb_trace(outdir: str, record: Dict[str, Any]) -> None:
    path = os.path.join(outdir, "ucb_trace.jsonl")
    _append_jsonl(path, record)

def _write_ucb_trace_doc(outdir: str) -> None:
    path = os.path.join(outdir, "ucb_trace.md")
    if os.path.exists(path):
        return
    os.makedirs(outdir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# UCB 参数变化记录\n\n"
            "本文件对应 pretrain 阶段的 UCB 参数轨迹，逐步追加到 ucb_trace.jsonl。\n\n"
            "字段说明：\n"
            "- i: 全局 step\n"
            "- phase: 阶段（pretrain）\n"
            "- t: UCB 迭代步数\n"
            "- alpha/l2/delta/S/d: UCB 超参数\n"
            "- A_inv: A 的逆矩阵（列表）\n"
            "- b: 向量 b\n"
            "- theta_hat: 估计参数向量\n"
            "- beta: 置信半径项\n"
        )


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
    solution_mode: Optional[str] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    logs: List[Dict[str, Any]] = []
    empty_agent_counts: Dict[str, int] = {}
    os.makedirs(outdir, exist_ok=True)
    progress_path = os.path.join(outdir, "progress.jsonl")
    progress_state_path = os.path.join(outdir, "progress_state.json")
    if phase == "pretrain":
        _write_ucb_trace_doc(outdir)

    # ✅ Test phase: disable selector updates to avoid label/data leakage (eval-only).
    if phase == "test":
        symphony_module.set_eval_only(True)

    # ✅ Cold_start mode: round-robin agent assignment (each task -> one agent)
    # If phase is "cold_start" and agents provided, use round-robin
    use_cold_start_round_robin = (phase == "cold_start" and agents is not None)

    for i, task in enumerate(tasks, start=start_index):
        t0 = time.time()
        task_obj = build_task_obj(task, i=i, requirements=requirements_override, solution_mode=solution_mode)
        bench = str(task.get("benchmark", "")).strip().lower()
        if print_each_step and bench in {"gsm8k", "gsm"} and i == start_index:
            prompt_preview = task_to_text(task)
            gold_preview = extract_gold_text(task)
            print("\n=== GSM8K SAMPLE DEBUG ===")
            print(f"task_id: {task.get('task_id') or task.get('id')}")
            print("PROMPT:\n", prompt_preview)
            print("GOLD_RAW:\n", gold_preview)
            print("==========================\n")

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
            raw_result: Any = None
            if selected_aid and selected_aid != "NA":
                agent_ids = [selected_aid]
            if selected_agent is None:
                final_text = ""
            else:
                # Retry cold_start calls on transient errors
                reqs = list(getattr(task_obj, "requirements", []) or ["analysis"])
                req0 = str(reqs[0]) if reqs else "analysis"
                raw_prompt = getattr(task_obj, "description", "") or task_to_text(task)
                legacy = {
                    "subtask_id": "1",
                    "steps": [
                        {"step_id": "1", "prompt": raw_prompt, "requirement": req0},
                    ],
                    "previous_results": [],
                    "original_problem": raw_prompt,
                    "final_result": "",
                    "user_id": "pretrain_cold_start",
                }
                last_err: Optional[str] = None
                raw_result: Any = None
                for attempt in range(3):
                    try:
                        result = selected_agent.execute_task(legacy)  # type: ignore[attr-defined]
                        raw_result = result
                        final_text = _unwrap_agent_result(result)
                        if final_text:
                            last_err = None
                            break
                        last_err = "empty_response"
                    except Exception as e:
                        last_err = f"{type(e).__name__}: {str(e)}"
                        raw_result = None

                    if attempt < 2:
                        time.sleep(1.5 * (attempt + 1))

                if last_err and not final_text:
                    final_text = f"[AGENT_ERROR] {last_err}"
                # keep last raw_result for debugging if needed
                _ = raw_result

            dt = time.time() - t0
            ok = is_success(final_text)
            if not final_text:
                for aid in agent_ids or [selected_aid]:
                    if aid:
                        empty_agent_counts[aid] = empty_agent_counts.get(aid, 0) + 1
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
            medical_extra: Dict[str, Any] = {}
            if str(task.get("benchmark", "")).strip().lower() == "medical_qa":
                raw = task.get("raw_data") or {}
                answer_idx = str(raw.get("answer_idx") or "").strip().upper()
                options = raw.get("options") if isinstance(raw.get("options"), dict) else None
                pred_token = _extract_mcq_token(pred) or _map_medical_qa_pred_to_token(task, pred)
                gold_token = _map_medical_qa_gold_to_token(task, gold_text)
                medical_extra = {
                    "answer_idx": answer_idx,
                    "pred_token": pred_token,
                    "gold_token": gold_token,
                    "options": options,
                }
            if ok == 0:
                acc = 0
            else:
                acc = is_correct(pred, gold_text, str(task.get("benchmark", "")), task=task)
            meta = task.get("scorer_metadata") if isinstance(task, dict) else None
            task_type = ""
            if isinstance(meta, dict):
                task_type = str(meta.get("task_name") or "").strip()
            logs.append(
                {
                    "i": i,
                    "phase": phase,
                    "task_id": task.get("task_id") or task.get("id"),
                    "benchmark": task.get("benchmark"),
                    "difficulty_bin": task.get("difficulty_bin"),
                    "task_type": task_type,
                    "agent_ids": agent_ids,
                    "subtask_count": subtask_count,
                    "subtask_meta": subtask_meta,
                    "ok": ok,
                    "pred_raw": pred_raw,
                    "raw_result": _to_list(raw_result),
                    "pred": pred,
                    "gold": gold_text,
                    "medical": medical_extra,
                    "acc": acc,
                    "latency_s": dt,
                }
            )
            if print_each_step:
                first_agent = agent_ids[0] if agent_ids else "NA"
                print(f"[{phase}] step={i} agent={first_agent} ok={ok} acc={acc} latency={dt:.2f}s")
            
            # ✅ Fix 2: Write progress_state.json for cold_start (resume support)
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
            medical_extra: Dict[str, Any] = {}
            if str(task.get("benchmark", "")).strip().lower() == "medical_qa":
                raw = task.get("raw_data") or {}
                answer_idx = str(raw.get("answer_idx") or "").strip().upper()
                options = raw.get("options") if isinstance(raw.get("options"), dict) else None
                gold_token = _map_medical_qa_gold_to_token(task, gold_text)
                medical_extra = {
                    "answer_idx": answer_idx,
                    "pred_token": "",
                    "gold_token": gold_token,
                    "options": options,
                }
            meta = task.get("scorer_metadata") if isinstance(task, dict) else None
            task_type = ""
            if isinstance(meta, dict):
                task_type = str(meta.get("task_name") or "").strip()
            logs.append(
                {
                    "i": i,
                    "phase": phase,
                    "task_id": task.get("task_id") or task.get("id"),
                    "benchmark": task.get("benchmark"),
                    "difficulty_bin": task.get("difficulty_bin"),
                    "task_type": task_type,
                    "agent_ids": [],
                    "ok": 0,
                    "pred": "",
                    "gold": gold_text,
                    "medical": medical_extra,
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
            # ✅ DEBUG: Print trace structure for first task to diagnose ok=0 issue
            if i == start_index:
                print(f"\n[DEBUG] === Trace structure for first task (step={i}) ===")
                print(f"[DEBUG] trace type: {type(trace)}, is_dict: {isinstance(trace, dict)}")
                traces_dbg = trace.get("traces", {}) or {}
                print(f"[DEBUG] traces type: {type(traces_dbg)}, is_dict: {isinstance(traces_dbg, dict)}, len: {len(traces_dbg) if isinstance(traces_dbg, dict) else 0}")
                if traces_dbg:
                    first = next(iter(traces_dbg.values()))
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
            if print_each_step:
                _print_trace_details(phase, i, trace)
            final_text, agent_ids = _extract_final_text_from_trace(trace)
            if not final_text and agent_ids:
                for aid in agent_ids:
                    if aid:
                        empty_agent_counts[aid] = empty_agent_counts.get(aid, 0) + 1
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
        medical_extra: Dict[str, Any] = {}
        if str(task.get("benchmark", "")).strip().lower() == "medical_qa":
            raw = task.get("raw_data") or {}
            answer_idx = str(raw.get("answer_idx") or "").strip().upper()
            options = raw.get("options") if isinstance(raw.get("options"), dict) else None
            pred_token = _extract_mcq_token(pred) or _map_medical_qa_pred_to_token(task, pred)
            gold_token = _map_medical_qa_gold_to_token(task, gold_text)
            medical_extra = {
                "answer_idx": answer_idx,
                "pred_token": pred_token,
                "gold_token": gold_token,
                "options": options,
            }
        if ok == 0:
            acc = 0
        else:
            acc = is_correct(pred, gold_text, str(task.get("benchmark", "")), task=task)
        if print_each_step and str(task.get("benchmark", "")).strip().lower() in {"gsm8k", "gsm"}:
            pred_norm = _extract_gsm8k_final(pred)
            gold_norm = _extract_gsm8k_final(gold_text)
            # 打印完整长度和末尾内容，用于诊断截断问题
            gold_len = len(gold_text) if gold_text else 0
            gold_tail = gold_text[-120:] if gold_text and len(gold_text) > 120 else gold_text
            print(f"[gsm8k][debug] pred_raw={repr(pred_raw)[:200]}")
            print(f"[gsm8k][debug] gold_raw_len={gold_len} gold_raw_tail={repr(gold_tail)}")
            print(f"[gsm8k][debug] pred_norm={pred_norm} gold_norm={gold_norm}")
        meta = task.get("scorer_metadata") if isinstance(task, dict) else None
        task_type = ""
        if isinstance(meta, dict):
            task_type = str(meta.get("task_name") or "").strip()
        logs.append(
            {
                "i": i,
                "phase": phase,
                "task_id": task.get("task_id") or task.get("id"),
                "benchmark": task.get("benchmark"),
                "difficulty_bin": task.get("difficulty_bin"),
                "task_type": task_type,
                "agent_ids": agent_ids,
                "subtask_count": subtask_count,
                "subtask_meta": subtask_meta,
                "ok": ok,
                "pred_raw": pred_raw,  # ✅ Raw output for debugging
                "trace_raw": _to_list(trace) if ok == 0 or acc == 0 else None,
                "pred": pred,  # ✅ Clean answer for evaluation (used in acc calculation)
                "gold": gold_text,
                "medical": medical_extra,
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
        # ✅ UCB trace (pretrain only)
        if phase == "pretrain":
            selector = getattr(symphony_module, "_global_orchestrator", None)
            selector = getattr(selector, "selector", None)
            if selector is not None:
                try:
                    theta_hat = getattr(selector, "theta_hat", [])
                    if callable(theta_hat):
                        theta_hat = theta_hat()
                    beta = getattr(selector, "beta", 0.0)
                    if callable(beta):
                        beta = beta()
                    rec = {
                        "i": i,
                        "phase": phase,
                        "t": int(getattr(selector, "t", 0)),
                        "d": int(getattr(selector, "d", 0)),
                        "alpha": float(getattr(selector, "alpha", 0.0)),
                        "l2": float(getattr(selector, "l2", 0.0)),
                        "delta": float(getattr(selector, "delta", 0.0)),
                        "S": float(getattr(selector, "S", 0.0)),
                        "A_inv": _to_list(getattr(selector, "A_inv", [])),
                        "b": _to_list(getattr(selector, "b", [])),
                        "theta_hat": _to_list(theta_hat),
                        "beta": float(beta),
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    }
                    _append_ucb_trace(outdir, rec)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Failed to record UCB trace: {e}")
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

    if empty_agent_counts:
        print(f"[{phase}] empty_response agents:")
        for aid, cnt in sorted(empty_agent_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {aid}: {cnt}")

    if phase == "test":
        symphony_module.set_eval_only(False)
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


def _collect_bbh_type_stats(all_logs: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[int]]]:
    stats: Dict[str, Dict[str, List[int]]] = {}
    for r in all_logs:
        if str(r.get("benchmark") or "").strip().lower() != "bbh":
            continue
        phase = str(r.get("phase") or "unknown")
        tname = str(r.get("task_type") or "unknown").strip() or "unknown"
        stats.setdefault(phase, {}).setdefault(tname, []).append(int(r.get("acc") or 0))
    return stats

def _macro_acc_from_type_stats(type_stats: Dict[str, List[int]]) -> Tuple[float, int, int]:
    if not type_stats:
        return 0.0, 0, 0
    per_type = []
    total_n = 0
    for vals in type_stats.values():
        total_n += len(vals)
        per_type.append(sum(vals) / max(1, len(vals)))
    k = len(per_type)
    return sum(per_type) / max(1, k), k, total_n

def write_bbh_macro_summary(all_logs: List[Dict[str, Any]], outdir: str) -> None:
    stats = _collect_bbh_type_stats(all_logs)
    if not stats:
        return
    rows: List[Dict[str, Any]] = []
    for phase, type_stats in sorted(stats.items()):
        for tname in sorted(type_stats.keys()):
            vals = type_stats[tname]
            rows.append(
                {
                    "phase": phase,
                    "task_type": tname,
                    "acc": sum(vals) / max(1, len(vals)),
                    "n": len(vals),
                    "macro_acc": "",
                    "k": "",
                }
            )
        macro, k, total_n = _macro_acc_from_type_stats(type_stats)
        rows.append(
            {
                "phase": phase,
                "task_type": "__macro__",
                "acc": "",
                "n": total_n,
                "macro_acc": macro,
                "k": k,
            }
        )
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "bbh_macro_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["phase", "task_type", "acc", "n", "macro_acc", "k"],
        )
        writer.writeheader()
        writer.writerows(rows)

def print_bbh_type_accuracy(all_logs: List[Dict[str, Any]]) -> None:
    stats = _collect_bbh_type_stats(all_logs)
    if not stats:
        return

    print("[BBH] macro acc by phase (avg over task types):")
    for phase in sorted(stats.keys()):
        macro, k, total_n = _macro_acc_from_type_stats(stats[phase])
        print(f"  - {phase}: macro_acc={macro:.4f} (k={k}, n={total_n})")

    # Detailed per-type accuracy for test phase
    test_stats = stats.get("test", {})
    if not test_stats:
        return
    print("[BBH] per-task-type accuracy (test):")
    for name in sorted(test_stats.keys()):
        vals = test_stats[name]
        acc = sum(vals) / max(1, len(vals))
        print(f"  - {name}: acc={acc:.4f} (n={len(vals)})")
    macro, k, total_n = _macro_acc_from_type_stats(test_stats)
    print(f"[BBH] macro test acc={macro:.4f} (k={k}, n={total_n})")

def _print_final_test_acc(all_logs: List[Dict[str, Any]]) -> None:
    if not all_logs:
        print("[ACC] test: N=0 acc=0.0000")
        return
    test_logs = [r for r in all_logs if str(r.get("phase")) == "test"]
    if not test_logs:
        print("[ACC] test: N=0 acc=0.0000")
        return
    total = len(test_logs)
    ok = sum(int(r.get("acc", r.get("ok", 0))) for r in test_logs)
    acc = ok / max(1, total)
    print(f"[ACC] test: N={total} acc={acc:.4f}")
    bbh_stats = _collect_bbh_type_stats(test_logs)
    if "test" in bbh_stats:
        macro, k, total_n = _macro_acc_from_type_stats(bbh_stats["test"])
        print(f"[ACC][BBH] test macro_acc={macro:.4f} (k={k}, n={total_n})")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-pool", type=str, required=True, help="Path to JSONL task pool")
    ap.add_argument("--benchmark", type=str, default=None, help="Filter tasks by benchmark (e.g., bbh)")
    ap.add_argument(
        "--bbh-task-types",
        type=str,
        default="",
        help="Comma-separated BBH task_name list to run (k<=23)",
    )
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
    ap.add_argument("--solution-mode", type=str, default="",
                    help="Inject SOLUTION_MODE into task prompt (e.g., Direct, ReAct, Synapse, Self-Consistency, Self-Refinement)")
    ap.add_argument("--outdir", type=str, default="pretrain_results")
    ap.add_argument("--resume-dir", type=str, default=None, help="Resume from existing outdir (uses progress_state.json)")
    ap.add_argument("--runtime-dir", type=str, default="runtime")
    ap.add_argument("--save-selector", type=str, default=None, help="Save UCB state after pretrain")
    ap.add_argument("--load-selector", type=str, default=None, help="Load UCB state and run test only")
    ap.add_argument("--plot-acc", action="store_true", help="Plot cumulative ACC curve")
    ap.add_argument("--print-cold-summary", action="store_true", help="Print cold_start summary accuracy")
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
        bench_tag = (str(args.benchmark) if args.benchmark else "all").replace("/", "_")
        n_tag = str(args.n) if args.n is not None else "all"
        agent_tag = _agent_tag_from_arg(args.agents, args.runtime_dir)
        name_tag = f"{bench_tag}_{agent_tag}_topL{args.topL}_plan{args.plan_k}_n{n_tag}"
        args.outdir = os.path.join(args.outdir, f"{date_str}_{name_tag}")

    bbh_task_types = [t.strip() for t in args.bbh_task_types.split(",") if t.strip()]
    tasks = load_tasks(
        args.task_pool,
        args.n,
        seed=args.seed,
        benchmark=args.benchmark,
        bbh_task_types=bbh_task_types if bbh_task_types else None,
    )
    if bbh_task_types:
        print(f"[BBH] selected task types (k={len(bbh_task_types)}): {', '.join(bbh_task_types)}")

    if args.val_n is not None:
        print("[WARN] --val-n is deprecated. Use --test-n instead.")
        args.test_n = int(args.val_n)

    if args.load_selector:
        args.cold_n = 0
        args.pretrain_n = 0

    total_needed = args.cold_n + args.pretrain_n + args.test_n
    if total_needed > len(tasks):
        args.test_n = max(0, len(tasks) - args.cold_n - args.pretrain_n)

    if (str(args.benchmark or "").strip().lower() == "bbh"):
        cold_tasks, pretrain_tasks, test_tasks = _stratified_split_bbh_by_phase(
            tasks,
            cold_n=args.cold_n,
            pretrain_n=args.pretrain_n,
            test_n=args.test_n,
            seed=args.seed,
        )
    else:
        cold_tasks = tasks[: args.cold_n]
        pretrain_tasks = tasks[args.cold_n : args.cold_n + args.pretrain_n]
        test_tasks = tasks[args.cold_n + args.pretrain_n : args.cold_n + args.pretrain_n + args.test_n]

    # load agents from command-line argument
    agent_ids = _parse_agent_ids(args.agents, args.runtime_dir)
    print(f"[INIT] resolved agent_ids: {agent_ids}")
    agents = load_agents_from_runtime(args.runtime_dir, agent_ids)
    resolved_keys: List[str] = []
    for ag in agents:
        symphony_module.register_agent(ag)
        aid = (
            str(getattr(ag, "agent_id", "")) or
            str(getattr(ag, "node_id", "")) or
            str(getattr(ag, "name", "")) or
            str(getattr(ag, "id", "")) or
            ""
        ).strip()
        if aid:
            resolved_keys.append(aid)
    if resolved_keys:
        print(f"[INIT] registered agents: {resolved_keys}")
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
    solution_mode = args.solution_mode.strip() if args.solution_mode else None

    if args.load_selector:
        # ✅ Fix 3: Initialize orchestrator before loading selector
        symphony_module.init(
            use_dynamic=True,
            topL=int(args.topL),
            linucb_alpha=float(args.ucb_alpha),
            linucb_l2=float(args.ucb_l2),
            plan_k=int(args.plan_k),
            use_planner_decompose=bool(args.planner_decompose),
        )
        # ✅ Fix 4: Re-register agents after init (in case init clears registry)
        for ag in agents:
            symphony_module.register_agent(ag)
        # Now load and freeze selector
        selector = load_selector(args.load_selector)
        symphony_module._global_orchestrator.selector = FrozenSelector(selector)
        symphony_module._global_orchestrator.use_dynamic = True
        if test_tasks:
            idx, logs = run_phase(
                "test",
                test_tasks,
                idx,
                args.cot_count,
                args.outdir,
                args.verbose,
                args.print_each_step,
                requirements_override=req_override,
                solution_mode=solution_mode,
            )
            all_logs.extend(logs)
    else:
        # cold start: static Top-L (no planner, no multi-CoT)
        symphony_module.init(
            use_dynamic=False,
            topL=1,  # ✅ 冷启动不使用 Top-L
            # plan_k=int(args.plan_k),
            plan_k=1,  # ✅ 冷启动不使用 planner
            use_planner_decompose=False,
        )
        # ✅ Fix 4: Re-register agents after init (in case init clears registry)
        for ag in agents:
            symphony_module.register_agent(ag)
        if cold_tasks:
            # idx, logs = run_phase("cold_start", cold_tasks, idx, args.cot_count, args.outdir, args.verbose, args.print_each_step, agents=agents) 
            idx, logs = run_phase(
                "cold_start",
                cold_tasks,
                idx,
                1,
                args.outdir,
                args.verbose,
                args.print_each_step,
                agents=agents,
                requirements_override=req_override,
                solution_mode=solution_mode,
            )  # ✅ C: cold_start 强制 cot_count=1
            all_logs.extend(logs)
            if args.print_cold_summary:
                cold_acc_sum = sum(1 for r in logs if int(r.get("acc", 0)) == 1)
                cold_total = len(logs)
                cold_acc = (cold_acc_sum / cold_total) if cold_total else 0.0
                print(f"[cold_start] summary acc={cold_acc:.4f} ({cold_acc_sum}/{cold_total})")

        # pretrain: Top-L + UCB (updates enabled)
        symphony_module.init(
            use_dynamic=True,
            topL=int(args.topL),
            linucb_alpha=float(args.ucb_alpha),
            linucb_l2=float(args.ucb_l2),
            plan_k=int(args.plan_k),
            use_planner_decompose=bool(args.planner_decompose),
        )
        # ✅ Fix 4: Re-register agents after init (in case init clears registry)
        for ag in agents:
            symphony_module.register_agent(ag)
        if pretrain_tasks:
            idx, logs = run_phase(
                "pretrain",
                pretrain_tasks,
                idx,
                args.cot_count,
                args.outdir,
                args.verbose,
                args.print_each_step,
                requirements_override=req_override,
                solution_mode=solution_mode,
            )
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
            idx, logs = run_phase(
                "test",
                test_tasks,
                idx,
                args.cot_count,
                args.outdir,
                args.verbose,
                args.print_each_step,
                requirements_override=req_override,
                solution_mode=solution_mode,
            )
            all_logs.extend(logs)

    if args.plot_acc:
        plot_acc_curves_by_phase(all_logs, args.outdir)
    write_accuracy_summary(all_logs, args.outdir)
    write_bbh_macro_summary(all_logs, args.outdir)
    print_bbh_type_accuracy(all_logs)
    _print_final_test_acc(all_logs)

    print(f"[OK] Pre-train done. outdir={args.outdir}")

if __name__ == "__main__":
    main()