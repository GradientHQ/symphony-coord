# models/base_loader.py
"""
Base model loader and API communication utilities.

This module provides functionality for loading and running large language models,
handling API communications, and task decomposition for complex problems.

Symphony 2.0 changes (important):
1) generate_task_dag now produces per-subtask capability tags (cap), so each step
   can have a different requirement (needed for Dynamic Beacon Selection to learn).
2) Lower temperature for JSON-structured outputs to reduce parse failures.
3) Make optional deps (vllm / pythonmonkey / regex) import-safe for reproducibility.
"""

from __future__ import annotations

import re
import subprocess
from typing import Dict, Any, List, Tuple, Optional, Union

# --- JSON loader: keep json5 if available, fallback to std json ---
try:
    import json5 as json  # type: ignore
except Exception:  # pragma: no cover
    import json  # type: ignore

# --- Optional deps (do NOT hard-crash if missing) ---
try:
    import regex as _regex  # noqa: F401
except Exception:  # pragma: no cover
    _regex = None

try:
    import pythonmonkey  # noqa: F401
except Exception:  # pragma: no cover
    pythonmonkey = None

try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None


# -------------------- helpers --------------------

def escape_backslashes_in_json(json_str: str) -> str:
    """
    Escape backslashes and newlines inside JSON string values.
    (Kept for backward compatibility; not always needed when using repair.js.)
    """

    def replacer(match):
        content = match.group(1)
        content = content.replace("\\", "\\\\")
        content = content.replace("\n", "\\n")
        return f"\"{content}\""

    return re.sub(r"\"([^\"]*?)\"", replacer, json_str, flags=re.DOTALL)


def call_api(
        prompt: str,
        system_prompt: Optional[str] = None,
        client: Optional["OpenAI"] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        logprobs: bool = False,
        top_logprobs: int = 1,
        **kwargs
) -> str:
    """
    Make API calls to OpenAI-compatible services (chat.completions).

    NOTE: This is independent from Symphony2.0 dynamic selection; kept as a utility.
    """
    if OpenAI is None:
        raise RuntimeError("openai python package not installed, cannot call_api().")

    if not client:
        assert api_key is not None, "Please input your api key"
        client = OpenAI(api_key=api_key, base_url=base_url)

    if not logprobs:
        top_logprobs = None

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    if prompt:
        messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        **kwargs
    )
    return response.choices[0].message.content


# -------------------- Symphony 2.0: capability taxonomy --------------------
# You should align these strings with your CapabilityManager / agents' capability names.
CAPABILITY_VOCAB: List[str] = [
    "mathematical-reasoning",
    "verification",
    "planning",
    "code-implementation",
    "debugging",
    "data-collection",
    "retrieval",
    "fact-checking",
    "summarization",
    "writing",
    "tool-use",
    "analysis",
]

# Synonym normalization for robustness (LLM may output variants)
_CAP_NORMALIZE: Dict[str, str] = {
    "math": "mathematical-reasoning",
    "reasoning": "mathematical-reasoning",
    "mathematics": "mathematical-reasoning",
    "verify": "verification",
    "validation": "verification",
    "check": "verification",
    "coding": "code-implementation",
    "implementation": "code-implementation",
    "programming": "code-implementation",
    "debug": "debugging",
    "search": "retrieval",
    "rag": "retrieval",
    "qa": "fact-checking",
    "factcheck": "fact-checking",
    "summary": "summarization",
    "summarize": "summarization",
}


def _normalize_cap(cap: str, fallback: str) -> str:
    """Normalize capability tag to a known vocabulary; fallback if unknown."""
    if not isinstance(cap, str) or not cap.strip():
        return _normalize_cap(fallback, "mathematical-reasoning")
    c = cap.strip()
    low = c.lower()
    if low in _CAP_NORMALIZE:
        return _CAP_NORMALIZE[low]
    # accept exact known tokens
    if c in CAPABILITY_VOCAB:
        return c
    # accept close variants like "fact_checking" -> "fact-checking"
    c2 = low.replace("_", "-")
    if c2 in CAPABILITY_VOCAB:
        return c2
    # last resort
    fb = _CAP_NORMALIZE.get(str(fallback).lower(), fallback)
    return fb if fb in CAPABILITY_VOCAB else "mathematical-reasoning"


def _extract_json_object(text: str) -> Optional[str]:
    """
    Best-effort extraction of a top-level JSON object from noisy model output.
    """
    if not isinstance(text, str):
        return None
    s = text.strip().lstrip("\ufeff")
    if s.startswith("{") and s.endswith("}"):
        return s

    # Try to capture the first {...} block (greedy but bounded by last brace)
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return s[start:end + 1]
    return None


def _repair_json_with_node(raw_text: str) -> Optional[str]:
    """
    Use repair.js (node) to fix malformed JSON.
    Assumes repair.js exists in project root (same as your original code).
    """
    try:
        proc = subprocess.run(
            ["node", "repair.js"],
            input=raw_text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        out = proc.stdout.decode("utf-8", errors="ignore").strip()
        if out:
            return out
        return None
    except Exception:
        return None


# -------------------- BaseModel --------------------

class BaseModel:
    """
    Base model class for loading and running large language models.

    Uses vLLM when available. If vLLM is not installed, raises a clear error.
    """

    def __init__(self, model_path: str, system_prompt: str = "", device: Optional[str] = None):
        self.device = device
        self.model_path = model_path

        if LLM is None or SamplingParams is None:
            raise RuntimeError(
                "vllm is not installed or failed to import. "
                "Install vllm to use BaseModel local inference."
            )

        # NOTE: vLLM will pick GPU automatically; device string is kept for logs.
        import os

        # ✅ vLLM 预热 256 seq 会 OOM：把 max_num_seqs / util / model_len 做成可配置
        gpu_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.75"))
        max_num_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS", "8"))
        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "2048"))

        # ✅ 尽量把 device 绑定到指定 GPU（最稳还是靠 CUDA_VISIBLE_DEVICES）
        # device 形如 "cuda:0"
        if device and isinstance(device, str) and device.startswith("cuda:"):
            try:
                gpu_idx = device.split(":")[-1].strip()
                if gpu_idx.isdigit():
                    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
            except Exception:
                pass

        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_util,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
        )

        print(f"Running on {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
              f" | util={gpu_util} max_num_seqs={max_num_seqs} max_model_len={max_model_len}")

        # kept for backward compatibility (some code may prepend it manually elsewhere)
        self.system_prompt = system_prompt.strip() + "\n" if system_prompt else ""

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.6,
            top_p: float = 0.9
    ) -> str:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text.strip()

    # -------------------------------------------------------------------------
    # Symphony 2.0: per-subtask capability tags for dynamic routing
    # -------------------------------------------------------------------------
    def generate_task_dag(
            self,
            task_background: str,
            task_question: str,
            user_input: str,
            requirement: str
    ) -> Tuple[Dict[str, List[str]], bool]:
        """
        Generate a DAG of subtasks.

        Symphony 2.0 change:
        - Instead of assigning the SAME `requirement` to every step,
          the model outputs a `cap` per subtask.
        - steps[str(i)] = [question_text, cap]
        """

        allowed_caps = ", ".join(CAPABILITY_VOCAB)
        prompt = f"""
You are a problem decomposer, not a solver.

Your goal:
Break the INPUT problem into a sequence of strictly computable subtasks.

CRITICAL:
- Do NOT solve the problem.
- Do NOT output any intermediate or final answer.
- Output ONLY valid JSON (no extra text).

Each subtask MUST be an object with:
- "q": the subtask question (string)
- "cap": a capability tag chosen from this closed set:
  [{allowed_caps}]

If unsure which cap to use, pick the closest one.
The "cap" must be a single string from the allowed set.

Output schema (STRICT):
{{
  "original_question": "<repeat the original question>",
  "subtasks": [
    {{"q": "Q1: ...", "cap": "mathematical-reasoning"}},
    {{"q": "Q2: ...", "cap": "verification"}}
  ]
}}

INPUT:
{user_input}

OUTPUT:
""".strip()

        # Lower temperature for structured JSON output stability
        result = self.generate(prompt, max_new_tokens=768, temperature=0.2, top_p=0.9)
        print(f"Raw result: {result}")
        print(f"[DEBUG] repr: {repr(result)}")

        try:
            cleaned = result.strip().lstrip("\ufeff")
            json_text = _extract_json_object(cleaned) or cleaned
            dag_dict = json.loads(json_text)

        except Exception as e1:
            print(f"[WARN] JSON decode failed: {e1}")
            # Try extracting JSON then repairing
            extracted = _extract_json_object(result)
            if extracted is None:
                print("[ERROR] JSON object not found in model output.")
                return {}, False

            repaired = _repair_json_with_node(extracted) or _repair_json_with_node(result)
            if repaired is None:
                print("[ERROR] repair.js failed to repair JSON.")
                return {}, False

            try:
                dag_dict = json.loads(repaired)
            except Exception as e2:
                print(f"[ERROR] JSON decode still failed after repair: {e2}")
                return {}, False

        # Build steps: { "1": [q, cap], ... }
        steps: Dict[str, List[str]] = {}
        raw_subtasks = dag_dict.get("subtasks", [])

        # Backward compatibility: if old format ["Q1", "Q2", ...]
        if isinstance(raw_subtasks, list) and raw_subtasks and isinstance(raw_subtasks[0], str):
            for idx, s in enumerate(raw_subtasks):
                cap = _normalize_cap(requirement, requirement)
                steps[str(idx + 1)] = [str(s), cap]
            return steps, True

        if not isinstance(raw_subtasks, list):
            print("[ERROR] 'subtasks' is not a list.")
            return {}, False

        for idx, item in enumerate(raw_subtasks):
            if isinstance(item, dict):
                q = str(item.get("q", "")).strip()
                cap_raw = str(item.get("cap", requirement)).strip()
                cap = _normalize_cap(cap_raw, requirement)
                if not q:
                    continue
                steps[str(idx + 1)] = [q, cap]
            else:
                # Unexpected item type; fallback
                q = str(item).strip()
                if q:
                    cap = _normalize_cap(requirement, requirement)
                    steps[str(idx + 1)] = [q, cap]

        if not steps:
            print("[ERROR] No valid subtasks parsed.")
            return {}, False

        # Debug print
        for sid in steps:
            print(steps[sid])

        return steps, True

    def extract_task(self, user_input: str) -> Tuple[str, str, bool]:
        """
        Extract background information and question from a problem statement.

        Symphony 2.0 suggestion:
        - Lower temperature for stable JSON outputs.
        """
        prompt = """You are a text extractor. Your task is ONLY to separate the background information from the question in math problem statements.

Each input contains:
- Background: context, assumptions, formulas, constraints, and setup.
- Question: the final sentence or phrase that asks what needs to be found, calculated, or determined.

IMPORTANT RULES:
- DO NOT solve or explain anything.
- DO NOT rewrite or infer any missing values.
- DO NOT modify, simplify, or expand any math expressions.
- DO NOT perform any calculations.
- DO NOT guess or assume anything.
- Only CUT the question part from the background and return both.

Return your output in the following strict JSON format:
{{
  "background": "<only the setup or context>",
  "question": "<only the question sentence>"
}}

Now extract background and question from the following input.
Output ONLY the JSON.

Input:
{user_input}

Output:""".format(user_input=user_input.strip())

        result = self.generate(prompt, temperature=0.2, max_new_tokens=512, top_p=0.9)
        print(f"Raw Result: {result}")
        print(f"[DEBUG] repr: {repr(result)}")

        try:
            cleaned = result.strip().lstrip("\ufeff")
            json_text = _extract_json_object(cleaned) or cleaned
            data = json.loads(json_text)
            return data.get("background", ""), data.get("question", ""), True

        except Exception as e1:
            print(f"[WARN] Direct JSON decode failed: {e1}")

            extracted = _extract_json_object(result)
            if extracted is None:
                print("[ERROR] JSON not found in output.")
                return "", "", False

            repaired = _repair_json_with_node(extracted) or _repair_json_with_node(result)
            if repaired is None:
                print("[ERROR] repair.js failed to repair JSON.")
                return "", "", False

            try:
                data = json.loads(repaired)
                return data.get("background", ""), data.get("question", ""), True
            except Exception as e2:
                print(f"[ERROR] Failed to parse repaired JSON: {e2}")
                return "", "", False








