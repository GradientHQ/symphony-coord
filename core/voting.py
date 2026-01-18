# core/voting.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class PlanVoteResult:
    win_key: str
    keys: List[str]
    weights: List[float]
    score: Dict[str, float]


def plan_vote_and_update(
    *,
    plan_runs: List[Dict[str, Any]],
    selector: Optional[Any],
    policy_name: str,
    win_bonus: float = 0.5,
    # inject a function to extract final answer/key from text
    extract_fn: Optional[Callable[[str], Optional[str]]] = None,
    # which policies trigger bonus update
    bonus_policies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Plan-level voting + winner bonus update.

    plan_runs item expected fields:
      - "final_text": str
      - "plan_weight": float (optional, default 0.0)
      - "step_records": List[dict] (optional)
           each record may contain "x": List[float]

    Returns dict:
      {
        "win_key": str,
        "keys": List[str],
        "weights": List[float],
        "score": Dict[str,float],
      }

    Notes:
      - extract_fn is injected to avoid importing symphony in core.
        In symphony/main you pass: SymphonyOrchestrator._extract_final_from_text
      - winner bonus update:
        for winner plan only, for each step_record with list x:
          selector.update(x, win_bonus)
    """
    if bonus_policies is None:
        bonus_policies = ["linucb", "linucb_cold", "linucb_warm", "linucb_warm2"]

    keys: List[str] = []
    weights: List[float] = []

    for pr in plan_runs or []:
        txt = pr.get("final_text", "")
        txt = "" if txt is None else str(txt)

        if extract_fn is not None:
            try:
                k = extract_fn(txt)
            except Exception:
                k = None
        else:
            k = None

        key = (k if isinstance(k, str) and k.strip() else txt.strip())
        keys.append(key)

        try:
            w = float(pr.get("plan_weight", 0.0) or 0.0)
        except Exception:
            w = 0.0
        weights.append(w)

    # weighted vote
    score: Dict[str, float] = {}
    for k, w in zip(keys, weights):
        score[k] = score.get(k, 0.0) + float(w)

    if score:
        win_key = max(score.items(), key=lambda kv: kv[1])[0]
    else:
        win_key = keys[0] if keys else ""

    # winner bonus update (only for LinUCB-like policies)
    if selector is not None and policy_name in bonus_policies:
        for pr, k in zip(plan_runs or [], keys):
            if k != win_key:
                continue
            for rec in pr.get("step_records", []) or []:
                x = rec.get("x")
                if isinstance(x, list):
                    try:
                        selector.update(x, float(win_bonus))
                    except TypeError:
                        # in case selector.update signature differs
                        selector.update(x, win_bonus)

    return {"win_key": win_key, "keys": keys, "weights": weights, "score": score}

