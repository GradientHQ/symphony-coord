# core/risk_guard.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import re
from difflib import SequenceMatcher


@dataclass
class RiskDecision:
    risk_level: str                 # "low" | "medium" | "high"
    action: str                     # "return" | "rerun" | "human_review"
    risk_score: float               # 0~1
    signals: Dict[str, float]       # disagreement/stability/domain/history...
    note: str = ""                  # human-readable explanation


@dataclass
class RiskGuardConfig:
    # threshold
    medium_th: float = 0.45
    high_th: float = 0.70

    # how to rerun
    extra_cot: int = 2
    max_rerun_rounds: int = 1

    # weights
    w_disagree: float = 0.45
    w_unstable: float = 0.30
    w_domain: float = 0.20
    w_history: float = 0.05

    # high-risk domains (you can extend)
    high_risk_domains: Tuple[str, ...] = ("medical", "finance", "legal")


class RiskAwareGuard:
    """
    Post-voting risk control module (does not change voting logic).
    """

    def __init__(self, cfg: Optional[RiskGuardConfig] = None):
        self.cfg = cfg or RiskGuardConfig()

    # ------------------ public API ------------------

    def assess(
        self,
        *,
        task_context: Dict[str, Any],
        subtask: Dict[str, Any],
        cot_results: List[str],
        voted_text: str,
        history_risk: float = 0.0,   # optional: [0,1]
    ) -> RiskDecision:
        domain_risk = self._domain_risk(task_context, subtask)
        disagree = self._disagreement_score(cot_results)
        unstable = self._instability_score(cot_results, voted_text)

        # combine
        score = (
            self.cfg.w_disagree * disagree
            + self.cfg.w_unstable * unstable
            + self.cfg.w_domain * domain_risk
            + self.cfg.w_history * float(max(0.0, min(1.0, history_risk)))
        )
        score = float(max(0.0, min(1.0, score)))

        if score >= self.cfg.high_th:
            return RiskDecision(
                risk_level="high",
                action="human_review",
                risk_score=score,
                signals={
                    "disagreement": disagree,
                    "instability": unstable,
                    "domain_risk": domain_risk,
                    "history_risk": float(history_risk),
                },
                note="High risk: strong disagreement/instability or high-risk domain. Recommend human review.",
            )

        if score >= self.cfg.medium_th:
            return RiskDecision(
                risk_level="medium",
                action="rerun",
                risk_score=score,
                signals={
                    "disagreement": disagree,
                    "instability": unstable,
                    "domain_risk": domain_risk,
                    "history_risk": float(history_risk),
                },
                note="Medium risk: trigger extra CoT sampling then re-vote.",
            )

        return RiskDecision(
            risk_level="low",
            action="return",
            risk_score=score,
            signals={
                "disagreement": disagree,
                "instability": unstable,
                "domain_risk": domain_risk,
                "history_risk": float(history_risk),
            },
            note="Low risk: return voted result.",
        )

    # ------------------ signals ------------------

    def _extract_final(self, s: str) -> Optional[str]:
        if not isinstance(s, str):
            return None
        m = re.findall(r"\\boxed\{([^}]*)\}", s)
        if m:
            return m[-1].strip()
        m = re.search(r"(?:final\\s*answer|答案)\\s*[:：]\\s*([A-Za-z0-9\\-\\./]+)", s, re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"\\b([A-Ea-e])\\b(?![A-Za-z])", s)
        if m:
            return m.group(1).upper()
        m = re.findall(r"[-+]?\\d+(?:/\\d+)?(?:\\.\\d+)?", s)
        if m:
            return m[-1]
        return None

    def _disagreement_score(self, cot_results: List[str]) -> float:
        """
        0 = fully consistent; 1 = highly inconsistent.
        We compute:
          - if finals extractable: 1 - max_vote_share
          - else: 1 - avg textual similarity to the medoid
        """
        if not cot_results:
            return 1.0

        finals = [self._extract_final(r) for r in cot_results]
        finals_ok = [f for f in finals if f]

        if finals_ok:
            counts: Dict[str, int] = {}
            for f in finals_ok:
                counts[f] = counts.get(f, 0) + 1
            m = max(counts.values())
            return float(max(0.0, min(1.0, 1.0 - m / max(1, len(finals_ok)))))

        # fallback: similarity-based
        sims = []
        for i in range(len(cot_results)):
            for j in range(i + 1, len(cot_results)):
                a, b = cot_results[i], cot_results[j]
                sims.append(SequenceMatcher(None, a, b).ratio())
        if not sims:
            return 1.0
        return float(max(0.0, min(1.0, 1.0 - sum(sims) / len(sims))))

    def _instability_score(self, cot_results: List[str], voted_text: str) -> float:
        """
        0 = stable; 1 = unstable.
        Heuristics:
          - if voted final cannot be extracted => more unstable
          - if finals extractable but margin is low => unstable
        """
        vf = self._extract_final(voted_text)
        if not vf:
            return 0.7  # strong uncertainty proxy

        finals = [self._extract_final(r) for r in cot_results]
        finals_ok = [f for f in finals if f]
        if not finals_ok:
            return 0.6

        counts: Dict[str, int] = {}
        for f in finals_ok:
            counts[f] = counts.get(f, 0) + 1

        top = sorted(counts.values(), reverse=True)
        if len(top) == 1:
            return 0.0  # unanimous
        margin = (top[0] - top[1]) / max(1, sum(top))
        # margin small => unstable
        return float(max(0.0, min(1.0, 1.0 - margin * 3.0)))

    def _domain_risk(self, task_context: Dict[str, Any], subtask: Dict[str, Any]) -> float:
        """
        Domain-based risk prior:
          - medical/finance/legal => higher base risk
          - also keyword triggers inside description
        """
        domain = str((task_context or {}).get("domain", "")).lower()
        req = str(subtask.get("requirement", "")).lower()
        desc = str(subtask.get("description", "")).lower()
        text = " ".join([domain, req, desc])

        base = 0.0
        if any(d in domain for d in self.cfg.high_risk_domains):
            base = 0.6

        # keyword prior (extend as needed)
        kw = ["diagnos", "dose", "treat", "prescrib", "investment", "trade", "leverage", "loan", "legal", "lawsuit"]
        if any(k in text for k in kw):
            base = max(base, 0.7)

        return float(max(0.0, min(1.0, base)))
