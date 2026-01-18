# core/cold_start.py
from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable

from core.linucb_selector import build_x


# -------------------------
# Logs / Stats
# -------------------------

@dataclass
class WarmupLog:
    task_id: str
    agent_id: str
    benchmark: str
    difficulty_bin: str

    match_score: float
    x: List[float]

    ok_api: int
    error_type: Optional[str]
    error_message: Optional[str]

    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    raw_text: str
    parsed_ok: int
    final_answer: str
    valid: int
    abstain: int
    confidence: float

    reward: float


@dataclass
class WarmupAgentStats:
    agent_id: str
    n: int
    n_ok_api: int
    avg_reward: float
    avg_latency_s: float
    sum_total_tokens: int


# -------------------------
# Helpers
# -------------------------

def _get_task_prompt(task: Dict[str, Any]) -> str:
    # Your tasks: task["raw_data"]["prompt"] exists for humaneval
    raw = task.get("raw_data") or {}
    p = raw.get("prompt")
    if isinstance(p, str) and p.strip():
        return p.strip()

    # fallback
    for k in ("prompt", "instruction", "input", "question", "text"):
        v = task.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return json.dumps(task, ensure_ascii=False)


def _get_task_id(task: Dict[str, Any], idx: int) -> str:
    return str(task.get("task_id") or task.get("id") or f"warm_{idx}")


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _extract_agent_prior(agent: Any, difficulty_bin: str, benchmark: str = "") -> float:
    """
    ✅ Extract agent prior match_score with learned priors support.
    
    Priority:
    1. Learned prior (if agent.learned_priors exists and bucket matches)
    2. Config prior (match_simple/match_hard from agent config)
    3. Default 0.5
    
    Args:
        agent: Agent object
        difficulty_bin: "hard" or "simple"/"easy"/etc.
        benchmark: e.g. "humaneval", "gsm8k" (used for bucket matching)
    
    Returns:
        match_score in [0, 1]
    """
    # Step 1: Try learned prior if bucket is available
    if benchmark:
        bucket = f"{benchmark}|{difficulty_bin}"
        lp = getattr(agent, "learned_priors", None)
        if lp is None and hasattr(agent, "__dict__"):
            lp = agent.__dict__.get("learned_priors")
        if isinstance(lp, dict) and bucket in lp:
            learned_score = _safe_float(lp[bucket], 0.5)
            return max(0.0, min(1.0, learned_score))
    
    # Step 2: Fallback to config priors
    if difficulty_bin == "hard":
        ms = getattr(agent, "match_hard", None)
        if ms is None and hasattr(agent, "__dict__"):
            ms = agent.__dict__.get("match_hard")
        return max(0.0, min(1.0, _safe_float(ms, 0.5)))
    else:
        ms = getattr(agent, "match_simple", None)
        if ms is None and hasattr(agent, "__dict__"):
            ms = agent.__dict__.get("match_simple")
        return max(0.0, min(1.0, _safe_float(ms, 0.5)))


def _build_dynamic_state_for_coldstart(agent: Any, benchmark: str = "", difficulty_bin: str = "") -> Dict[str, Any]:
    """
    ✅ Cold start dynamic state with learned prior support for reputation.
    
    If learned_priors exist and bucket matches, inject prior_success into reputation.
    Otherwise use conservative defaults.
    
    Args:
        agent: Agent object
        benchmark: e.g. "humaneval", "gsm8k" (for bucket matching)
        difficulty_bin: "hard" or "simple"/"easy"/etc. (for bucket matching)
    
    Returns:
        Dynamic state dict with load, latency_ms, reputation
    """
    lat = getattr(agent, "base_latency_ms", None)
    if lat is None and hasattr(agent, "__dict__"):
        lat = agent.__dict__.get("base_latency_ms")
    latency_ms = _safe_float(lat, 500.0)
    
    # Try to inject learned prior into reputation if bucket is available
    reputation = 0.5  # default
    if benchmark and difficulty_bin:
        bucket = f"{benchmark}|{difficulty_bin}"
        lp = getattr(agent, "learned_priors", None)
        if lp is None and hasattr(agent, "__dict__"):
            lp = agent.__dict__.get("learned_priors")
        if isinstance(lp, dict) and bucket in lp:
            # Learned prior represents stability/format-success rate
            # Use it as reputation prior (experience-based prior)
            reputation = max(0.0, min(1.0, _safe_float(lp[bucket], 0.5)))

    return {
        "load": 0.0,
        "latency_ms": float(latency_ms),
        "reputation": float(reputation),
    }


def _parse_one_line_json(raw_text: str) -> Tuple[int, Dict[str, Any]]:
    """
    Agents are prompted to output exactly one line JSON.
    Return (parsed_ok, obj)
    """
    if not isinstance(raw_text, str):
        return 0, {}
    s = raw_text.strip()
    if not s:
        return 0, {}
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return 1, obj
        return 0, {}
    except Exception:
        return 0, {}


def _compute_reward_from_agent_json(
    *,
    ok_api: int,
    parsed_ok: int,
    obj: Dict[str, Any],
    latency_s: float,
    latency_penalty: float,
) -> float:
    """
    Warmup reward policy (format-first):
      - API fail => 0
      - JSON parse fail => 0
      - valid must be 1, abstain must be 0, final_answer non-empty => base 1, else 0
      - optional latency penalty: reward -= latency_penalty * sqrt(latency_norm)
    """
    if ok_api != 1 or parsed_ok != 1:
        return 0.0

    valid = int(obj.get("valid", 0) or 0)
    abstain = int(obj.get("abstain", 0) or 0)
    final_answer = obj.get("final_answer", "")

    base = 1.0 if (valid == 1 and abstain == 0 and str(final_answer).strip() != "") else 0.0
    if base <= 0.0:
        return 0.0

    if latency_penalty <= 0.0:
        return 1.0

    # Normalize latency to [0,1] with 2s scale (same as your on_task_result)
    lat_norm = min(1.0, max(0.0, float(latency_s) / 2.0))
    penal = float(latency_penalty) * (lat_norm ** 0.5)
    r = base - penal
    return max(0.0, min(1.0, r))


def _summarize_by_agent(logs: List[WarmupLog]) -> Dict[str, WarmupAgentStats]:
    agg: Dict[str, Dict[str, float]] = {}
    for r in logs:
        d = agg.setdefault(r.agent_id, {"n": 0.0, "n_ok": 0.0, "sum_r": 0.0, "sum_lat": 0.0, "sum_tok": 0.0})
        d["n"] += 1.0
        d["n_ok"] += float(r.ok_api)
        d["sum_r"] += float(r.reward)
        d["sum_lat"] += float(r.latency_s)
        d["sum_tok"] += float(r.total_tokens)

    out: Dict[str, WarmupAgentStats] = {}
    for aid, d in agg.items():
        n = int(d["n"])
        out[aid] = WarmupAgentStats(
            agent_id=aid,
            n=n,
            n_ok_api=int(d["n_ok"]),
            avg_reward=(d["sum_r"] / n) if n > 0 else 0.0,
            avg_latency_s=(d["sum_lat"] / n) if n > 0 else 0.0,
            sum_total_tokens=int(d["sum_tok"]),
        )
    return out


def build_priors_from_logs(logs: List[WarmupLog]) -> Dict[str, Dict[str, float]]:
    """
    ✅ Build learned priors from warmup logs (by bucket: benchmark|difficulty_bin).
    
    Returns:
        priors[agent_id][bucket] = avg_reward  (stability/format-success rate from warmup)
    
    Note:
        - bucket format: "benchmark|difficulty_bin"
        - Currently reward is format-first (stability), not accuracy
        - Can later upgrade to accuracy-based prior when correctness evaluator is added
    """
    # ✅ P0-B: Use routing.make_bucket() for consistent bucket rule
    try:
        from core.routing import make_bucket
    except ImportError:
        # Fallback: use same rule as routing.make_bucket()
        def make_bucket(benchmark: str, difficulty_bin: str) -> str:
            bench = str(benchmark or "").strip()
            diff = str(difficulty_bin or "").strip() or "unknown"
            return f"{bench}|{diff}"
    
    agg: Dict[Tuple[str, str], List[float]] = {}
    for r in logs:
        bucket = make_bucket(r.benchmark, r.difficulty_bin)  # ✅ Use unified bucket rule
        agg.setdefault((r.agent_id, bucket), []).append(float(r.reward))
    
    priors: Dict[str, Dict[str, float]] = {}
    for (aid, bucket), rs in agg.items():
        if rs:
            avg_reward = sum(rs) / len(rs)
            priors.setdefault(aid, {})[bucket] = max(0.0, min(1.0, avg_reward))
    
    return priors


# -------------------------
# Core: cold start
# -------------------------

def cold_start_warmup(
    *,
    tasks: List[Dict[str, Any]],
    agents_dict: Dict[str, Any],
    selector: Optional[Any] = None,
    n_warmup: int = 120,
    seed: int = 0,
    shuffle: bool = True,
    latency_scale_ms: float = 2000.0,
    latency_penalty: float = 0.2,   # align with on_task_result: reward - 0.2*sqrt(lat_norm)
) -> Tuple[List[WarmupLog], Dict[str, WarmupAgentStats]]:
    """
    Cold start for OpenRouter agents, aligned with your LinUCB(d=6) build_x.

    - Round-robin assign warmup tasks across agents.
    - Call OpenRouter with generate_with_metadata (token usage + structured errors).
    - Parse strict JSON output.
    - Compute reward (format-success oriented by default).
    - Warm-start selector: selector.update(x, reward) if selector is not None.

    Returns (logs, stats_by_agent).
    """
    rng = random.Random(seed)
    agent_ids = list(agents_dict.keys())
    if not agent_ids:
        raise ValueError("agents_dict is empty")

    warm_tasks = list(tasks[: max(0, int(n_warmup))])
    if shuffle:
        rng.shuffle(warm_tasks)

    logs: List[WarmupLog] = []

    for i, task in enumerate(warm_tasks):
        aid = agent_ids[i % len(agent_ids)]
        agent = agents_dict[aid]

        task_id = _get_task_id(task, i)
        bench = str(task.get("benchmark", ""))
        diff = str(task.get("difficulty_bin", "")) or "unknown"

        prompt = _get_task_prompt(task)

        # ✅ build x using learned priors (if available) or config priors + cold dynamic defaults
        # Pass benchmark to enable learned prior lookup by bucket
        match_score = _extract_agent_prior(agent, diff, bench)
        ds = _build_dynamic_state_for_coldstart(agent, bench, diff)
        x = build_x(match_score=match_score, dynamic_state=ds, available=True, latency_scale_ms=float(latency_scale_ms))

        # --- call agent ---
        t0 = time.time()
        ok_api = 0
        meta: Dict[str, Any] = {}
        raw_text = ""

        bm = getattr(agent, "base_model", None)
        if bm is None:
            ok_api = 0
            meta = {"success": False, "error_type": "no_base_model", "error_message": "agent.base_model is None"}
        else:
            try:
                if hasattr(bm, "generate_with_metadata"):
                    meta = bm.generate_with_metadata(
                        prompt,
                        max_tokens=int(getattr(agent, "max_tokens", 512)),
                        temperature=float(getattr(agent, "temperature", 0.2)),
                        top_p=float(getattr(agent, "top_p", 0.9)),
                    )
                    ok_api = 1 if meta.get("success", False) else 0
                    raw_text = (meta.get("response", "") or "").strip()
                else:
                    # fallback
                    raw_text = str(bm.generate(prompt)).strip()
                    ok_api = 1
                    meta = {"success": True, "response": raw_text}
            except Exception as e:
                ok_api = 0
                meta = {"success": False, "error_type": "exception", "error_message": repr(e)}
                raw_text = ""

        latency_s = time.time() - t0

        # --- parse strict json output ---
        parsed_ok, obj = _parse_one_line_json(raw_text)

        final_answer = str(obj.get("final_answer", "") or "")
        valid = int(obj.get("valid", 0) or 0)
        abstain = int(obj.get("abstain", 0) or 0)
        confidence = _safe_float(obj.get("confidence", 0.0), 0.0)

        reward = _compute_reward_from_agent_json(
            ok_api=ok_api,
            parsed_ok=parsed_ok,
            obj=obj,
            latency_s=latency_s,
            latency_penalty=float(latency_penalty),
        )

        # --- warm-start selector ---
        if selector is not None:
            try:
                selector.update(x, reward)
            except Exception:
                # do not crash warmup
                pass

        logs.append(
            WarmupLog(
                task_id=task_id,
                agent_id=aid,
                benchmark=bench,
                difficulty_bin=diff,
                match_score=float(match_score),
                x=x,
                ok_api=int(ok_api),
                error_type=meta.get("error_type"),
                error_message=meta.get("error_message"),
                latency_s=float(latency_s),
                prompt_tokens=int(meta.get("prompt_tokens", 0) or 0),
                completion_tokens=int(meta.get("completion_tokens", 0) or 0),
                total_tokens=int(meta.get("total_tokens", 0) or 0),
                raw_text=raw_text,
                parsed_ok=int(parsed_ok),
                final_answer=final_answer,
                valid=int(valid),
                abstain=int(abstain),
                confidence=float(confidence),
                reward=float(reward),
            )
        )

    stats = _summarize_by_agent(logs)
    return logs, stats


def dump_jsonl(logs: List[WarmupLog], path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        for r in logs:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def dump_priors(priors: Dict[str, Dict[str, float]], path: str) -> None:
    """
    ✅ Dump learned priors to JSON file for later loading.
    
    Args:
        priors: priors[agent_id][bucket] = avg_reward
        path: Output JSON file path
    """
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2, ensure_ascii=False)


def load_priors(path: str) -> Dict[str, Dict[str, float]]:
    """
    ✅ Load learned priors from JSON file.
    
    Args:
        path: Input JSON file path
    
    Returns:
        priors[agent_id][bucket] = avg_reward, or empty dict if file not found
    """
    import json
    import os
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
