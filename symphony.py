"""
Symphony 2.0 core execution engine (Path-1: centralized orchestrator).

✅ Symphony 2.0 features:
- Two-stage agent selection:
    Stage-1: Top-L by static capability match_score (Symphony 1.0 compatible)
    Stage-2: Global LinUCB selects within Top-L using dynamic_state features
- Multi-CoT execution and voting per subtask
- Online update after voting (winner bonus + latency penalty)
- Optional Symphony 1.0-style planning decomposition (multiple planners produce chains)

✅ This patched version additionally:
- Planner branch also performs online updates (closes the loop in planner mode)
- Planner weighted vote keys on extracted Final (not whole raw text)
- Never uses x[1] as match_score (build_x may normalize -> x[1] != raw match)
- Safer fallback: do not pick unavailable agent when pool is empty
- Optional correctness reward if gold label is provided in task/subtask context

Return modes:
- "aggregate": returns a multi-subtask report (string)
- "final": returns final answer only (string, BBH-friendly)
- "trace": returns dict with per-run traces (for debugging / saving)
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------- imports (package / local) ----------------------
try:
    # Package mode
    from symphony.protocol.task_contract import Task  # type: ignore
    from symphony.agents.agent import Agent  # type: ignore
    from symphony.core.linucb_selector import GlobalLinUCB, build_x  # type: ignore
except Exception:  # pragma: no cover
    # Local mode
    from protocol.task_contract import Task  # type: ignore
    from agents.agent import Agent  # type: ignore
    from core.linucb_selector import GlobalLinUCB, build_x  # type: ignore

# Risk guard is optional
try:
    from core.risk_guard import RiskAwareGuard, RiskGuardConfig  # type: ignore
except Exception:  # pragma: no cover
    RiskAwareGuard = None  # type: ignore
    RiskGuardConfig = None  # type: ignore


# ---------------------- orchestrator ----------------------
class SymphonyOrchestrator:
    """Main orchestrator for multi-agent task execution (Path-1)."""

    def __init__(
            self,
            verbose: bool = False,
            # ---- Dynamic Beacon Selection knobs (2.0) ----
            use_dynamic: bool = True,
            topL: int = 3,
            linucb_alpha: float = 1.0,
            linucb_l2: float = 1.0,
            latency_scale_ms: float = 2000.0,
            latency_penalty: float = 0.2,
            win_bonus: float = 0.5,
            # ---- Optional correctness reward (for better scores) ----
            correctness_bonus: float = 0.0,  # add when voted final matches gold
            incorrect_penalty: float = 0.0,  # subtract when voted final mismatches gold
            # ---- Optional Symphony 1.0 planner ----
            plan_k: int = 1,
            # ---- Use planner to decompose even when plan_k == 1 ----
            use_planner_decompose: bool = False,
            # ---- Risk Guard ----
            enable_risk_guard: bool = False,
            # ---- Shared Blackboard / by-ref dispatch (Path-2 style, optional) ----
            dispatch_mode: str = "local",  # "local" | "shared_bb"
            shared_timeout_s: float = 30.0,  # wait result timeout
            shared_poll_interval: float = 0.01,
            requester_id: Optional[str] = None,  # pick requester agent by id if provided
            # ---- ✅ P0-1: Cold-start priors injection ----
            priors: Optional[Dict[str, Dict[str, float]]] = None,  # agent_id -> bucket -> prior
            priors_path: Optional[str] = None,  # path to priors JSON file
            # ---- ✅ P0-3: Strict routing mode (experiment mode) ----
            strict_routing: bool = False,  # If True, routing failure raises instead of fallback
            # ---- ✅ Eval-only (no selector updates; avoid test-phase leakage) ----
            eval_only: bool = False,

    ) -> None:
        self.lock = threading.Lock()
        self.verbose = bool(verbose)
        self.eval_only = bool(eval_only)

        # agent registry
        self.agents: List[Agent] = []

        # dynamic selection knobs
        self.use_dynamic = bool(use_dynamic)
        self.topL = max(1, int(topL))
        self.latency_scale_ms = float(latency_scale_ms)
        self.latency_penalty = float(latency_penalty)
        self.win_bonus = float(win_bonus)

        # correctness reward knobs
        self.correctness_bonus = float(correctness_bonus)
        self.incorrect_penalty = float(incorrect_penalty)

        # planner knobs (Symphony 1.0 style)
        self.plan_k = max(1, int(plan_k))
        self.use_planner_decompose = bool(use_planner_decompose)

        # optional risk guard
        self.enable_risk_guard = bool(enable_risk_guard)
        self.risk_guard = None
        if self.enable_risk_guard and RiskAwareGuard is not None and RiskGuardConfig is not None:
            self.risk_guard = RiskAwareGuard(RiskGuardConfig())

        # ✅ Global LinUCB (single global A,b)
        # build_x returns a vector (commonly 6-dim): [1, match, load, lat_norm, rep, available]
        self.selector: Optional[GlobalLinUCB] = None
        if self.use_dynamic:
            self.selector = GlobalLinUCB(d=6, l2=float(linucb_l2), alpha=float(linucb_alpha))
        # ---- shared dispatch knobs ----
        self.dispatch_mode = str(dispatch_mode or "local")
        self.shared_timeout_s = float(shared_timeout_s)
        self.shared_poll_interval = float(shared_poll_interval)
        self.requester_id = requester_id
        self.strict_routing = bool(strict_routing)  # ✅ P0-3: Strict routing mode
        
        # ✅ P0-1: Load and inject learned priors
        _priors: Dict[str, Dict[str, float]] = {}
        if priors is not None:
            _priors = priors
        elif priors_path:
            try:
                from core.cold_start import load_priors
                _priors = load_priors(priors_path)
                if self.verbose:
                    self._log(f"[ORCHESTRATOR] Loaded priors from {priors_path} ({len(_priors)} agents)")
            except Exception as e:
                if self.verbose:
                    self._log(f"[WARN] Failed to load priors from {priors_path}: {e}")
        
        # Store priors for later injection (will be injected when agents are registered)
        self._learned_priors = _priors

    # ---------------------- logging ----------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ---------------------- lifecycle ----------------------
    def register_agent(self, agent: Agent) -> None:
        with self.lock:
            if agent not in self.agents:
                self.agents.append(agent)
                # ✅ P0-1: Use unified agent key resolution
                agent_key = self._resolve_agent_key(agent)
                
                # ✅ P0-1: Inject learned priors into agent (with logging)
                if agent_key and agent_key in self._learned_priors:
                    agent.learned_priors = self._learned_priors[agent_key]
                    bucket_count = len(self._learned_priors[agent_key])
                    self._log(f"[PRIORS] ✅ Injected agent={agent_key} buckets={bucket_count}")
                elif self._learned_priors and agent_key:
                    # ✅ P0-1: Log miss (only first few to avoid spam)
                    available_keys = list(self._learned_priors.keys())[:5]
                    if not hasattr(self, "_priors_miss_logged"):
                        self._priors_miss_logged = set()
                    if agent_key not in self._priors_miss_logged and len(self._priors_miss_logged) < 5:
                        self._priors_miss_logged.add(agent_key)
                        self._log(f"[PRIORS] ❌ Miss agent={agent_key} available_keys={available_keys}")
                
                self._log(f"[ORCHESTRATOR] Registered agent: {agent_key}")

    def get_registered_agents(self) -> List[Agent]:
        return list(self.agents)
    
    # ---------------------- P0-1: Unified agent key resolution ----------------------
    @staticmethod
    def _resolve_agent_key(agent: Agent) -> str:
        """
        ✅ P0-1: Resolve agent key for priors lookup.
        
        Priority chain: agent_id -> node_id -> name -> id
        
        Args:
            agent: Agent object
        
        Returns:
            Agent key string (empty if none found)
        """
        aid = (
            str(getattr(agent, "agent_id", "")) or
            str(getattr(agent, "node_id", "")) or
            str(getattr(agent, "name", "")) or
            str(getattr(agent, "id", "")) or
            ""
        ).strip()
        return aid

    # ---------------------- helpers: requirement normalization ----------------------
    @staticmethod
    def _normalize_requirement(req: str) -> str:
        """
        Make matching slightly more robust:
        - lowercase
        - spaces/hyphens -> underscores
        """
        r = (req or "").strip().lower()
        r = re.sub(r"[\s\-]+", "_", r)
        return r

    # ---------------------- helpers: gold/correctness ----------------------
    def _get_gold_from_context(self, ctx: Dict[str, Any]) -> Union[None, str, List[str]]:
        """
        Look for gold label in context. Supported:
          ctx["gold"] = "yes" or "A" or "42"
          ctx["gold"] = ["A", "B"]  (multiple acceptable)
        """
        if not isinstance(ctx, dict):
            return None
        gold = ctx.get("gold", None)
        if gold is None:
            return None
        if isinstance(gold, str):
            g = gold.strip()
            return g if g else None
        if isinstance(gold, (list, tuple, set)):
            out: List[str] = []
            for x in gold:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out if out else None
        return None

    def _canon_answer(self, s: str) -> str:
        """
        Canonicalize an answer for comparison:
        - try extract_final first
        - strip, collapse spaces, lowercase for yes/no/valid/invalid, keep letter uppercase
        """
        t = (s or "").strip()
        fin = self._extract_final_from_text(t)
        t = (fin or t).strip()
        t = re.sub(r"\s+", " ", t)

        # normalize common BBH labels
        low = t.lower()
        if low in ("yes", "no", "valid", "invalid", "true", "false"):
            return low
        if re.fullmatch(r"[A-Za-z]", t):
            return t.upper()
        return t

    def _is_correct(self, pred_text: str, gold: Union[str, List[str], None]) -> Optional[bool]:
        if gold is None:
            return None
        pred = self._canon_answer(pred_text)
        if isinstance(gold, str):
            return pred == self._canon_answer(gold)
        if isinstance(gold, list):
            gold_set = {self._canon_answer(x) for x in gold if isinstance(x, str)}
            return pred in gold_set if gold_set else None
        return None
    
    # ---------------------- P0-0: Unified feature building helper ----------------------
    def _build_x_from_candidate_or_fallback(
        self,
        candidate: Dict[str, Any],
        agent: Agent,
        dynamic_state: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        ✅ P0-0: Unified helper to build x from candidate (ensures ms=sim_emb, rep=prior_success).
        
        This ensures all branches (dynamic, non-dynamic, risk rerun, planner) use consistent
        feature definitions: ms = sim_emb, rep = prior_success.
        
        Args:
            candidate: Candidate dict from TopL with {"agent", "sim_emb", "prior_success", ...}
            agent: Agent object (may differ from candidate["agent"] if using fallback)
            dynamic_state: Optional dynamic state dict (if None, will fetch from agent)
        
        Returns:
            Feature vector x [1, ms, load, lat, rep, av]
            where ms = sim_emb, rep = prior_success
        """
        # Try routing.build_x_from_candidate() first (preferred path)
        try:
            from core.routing import build_x_from_candidate
            x, _, _ = build_x_from_candidate(
                candidate=candidate,
                agent=agent,
                latency_scale_ms=self.latency_scale_ms,
            )
            return x
        except Exception:
            # Fallback: build x manually with consistent feature definitions
            pass
        
        # ✅ P0-C: Extract sim_emb and prior_success from candidate (required)
        # ✅ P0-C: sim_emb fallback must be neutral (0.5), NOT match_score (composite)
        sim_emb_val = float(candidate.get("sim_emb", 0.5))  # Do NOT use match_score as fallback
        prior_success_val = float(candidate.get("prior_success", 0.5))
        
        # Get dynamic state if not provided
        if dynamic_state is None:
            dynamic_state = self._agent_state(agent)
        
        # ✅ P0-0: Build x with consistent features: ms = sim_emb, rep = prior_success
        x = build_x(
            match_score=sim_emb_val,  # ✅ ms = sim_emb (not composite match_score)
            dynamic_state={
                "load": float(dynamic_state.get("load", 0.0)),
                "latency_ms": float(dynamic_state.get("latency_ms", 500.0)),
                "reputation": prior_success_val,  # ✅ rep = prior_success (not stt.reputation)
            },
            available=bool(dynamic_state.get("available", True)),
            latency_scale_ms=self.latency_scale_ms,
        )
        return x

    # ---------------------- public entry ----------------------
    def execute_task(
            self,
            task: Task,
            cot_count: int = 3,
            return_mode: str = "aggregate",  # "aggregate" | "final" | "trace"
    ) -> Any:
        """
        Main execution:
        - If plan_k > 1: planning => execute each plan chain => weighted vote.
        - Else: decompose by task.requirements => execute each subtask with multi-CoT voting.
        """
        task_text = getattr(task, "description", "") or ""
        ctx = getattr(task, "context", {}) or {}
        # ✅ stable run_id for this top-level task (used in shared_bb mode)
        if isinstance(ctx, dict) and "_run_id" not in ctx:
            ctx["_run_id"] = f"run:{uuid.uuid4().hex[:8]}"
            try:
                task.context = ctx  # type: ignore[attr-defined]
            except Exception:
                pass

        if not self.agents:
            if return_mode == "trace":
                return {"error": "No agents registered", "results": {}, "traces": {}}
            return "[ERROR] No agents registered"

        # ---------- (A) Optional planner mode (Symphony 1.0-style, patched to 2.0 loop) ----------
        if self.plan_k > 1 or self.use_planner_decompose:
            m = self.plan_k if self.plan_k > 1 else 1
            plans = self._plan_chains_v1(task_text=task_text, ctx=ctx, m=m)
            plan_answers: List[str] = []
            plan_weights: List[float] = []
            plan_traces: List[Dict[str, Any]] = []

            for p in plans:
                ans, w, tr = self._run_plan_chain_v1(base_task=task_text, chain=p["chain"], base_ctx=ctx)
                plan_answers.append(ans)
                plan_weights.append(w)
                plan_traces.append({"planner": p.get("planner", ""), "w": w, "trace": tr, "chain": p["chain"]})

            # ✅ vote on extracted final keys (not whole raw text)
            plan_keys = [(self._extract_final_from_text(a) or str(a).strip()) for a in plan_answers]
            win_key = self._weighted_vote(plan_keys, plan_weights)

            # choose representative full text with max weight among those sharing win_key
            best_i = 0
            best_w = -1e18
            for i, (k, w) in enumerate(zip(plan_keys, plan_weights)):
                if k == win_key and float(w) > best_w:
                    best_w = float(w)
                    best_i = i
            final_text = plan_answers[best_i] if plan_answers else ""

            # ✅ optional correctness reward at planner-level
            gold = self._get_gold_from_context(ctx)
            correct = self._is_correct(win_key, gold)

            # ✅ winner-bonus updates for all steps in winning trajectory(ies)
            # Skip updates in eval_only (e.g. test phase) to avoid label/data leakage.
            if self.use_dynamic and self.selector is not None and not self.eval_only:
                for i, k in enumerate(plan_keys):
                    if k != win_key:
                        continue
                    tr = plan_traces[i].get("trace", {}) if isinstance(plan_traces[i], dict) else {}
                    recs = tr.get("records", [])
                    for rec in recs:
                        x = rec.get("x")
                        if isinstance(x, list):
                            bonus = float(self.win_bonus)
                            if correct is True:
                                bonus += float(self.correctness_bonus)
                            elif correct is False:
                                bonus -= float(self.incorrect_penalty)
                            self.selector.update(x, bonus)

            if return_mode == "trace":
                # Build traces dict compatible with non-planner mode for pretrain.py
                # Extract runs from winning plan's records to build a compatible trace structure
                traces_dict: Dict[str, Any] = {}
                all_runs: List[Dict[str, Any]] = []
                
                # Extract runs from winning plan's trace (records are the run_records)
                if plan_traces and best_i < len(plan_traces):
                    winning_trace = plan_traces[best_i].get("trace", {})
                    if isinstance(winning_trace, dict):
                        # _run_plan_chain_v1 returns {"steps": [...], "records": [...]}
                        records = winning_trace.get("records", [])
                        if isinstance(records, list):
                            # records are the run_records from plan chain execution
                            all_runs = records
                
                # Build a single trace entry compatible with pretrain.py (expects {"traces": {"sub_1": {...}}})
                if all_runs or final_text:
                    traces_dict["sub_1"] = {
                        "requirement": "planning",
                        "context": ctx,
                        "gold": gold,
                        "vote_count": dict(Counter(plan_keys)),
                        "vote_weight_by_match_score": {k: w for k, w in zip(plan_keys, plan_weights)},
                        "correct": correct,
                        "runs": all_runs,
                        "voted": final_text,
                        "voted_final": win_key,
                    }
                
                return {
                    "results": {"sub_1": final_text} if final_text else {},
                    "traces": traces_dict,
                    "final": win_key,
                    "final_text": final_text,
                    "answers": plan_answers,
                    "keys": plan_keys,
                    "weights": plan_weights,
                    "plans": plan_traces,
                    "gold": gold,
                    "correct": correct,
                }

            if return_mode == "final":
                return (win_key or (self._extract_final_from_text(final_text) or final_text)).strip()

            # aggregate
            rep = "## Symphony Planner Result\n\n"
            rep += f"**Original Task**: {task_text}\n\n"
            for i, (a, w) in enumerate(zip(plan_answers, plan_weights), 1):
                rep += f"{i}. (w={w:.3f}) {a}\n\n"
            rep += f"\n**Final answer**: {win_key}\n"
            return rep.strip()

        # ---------- (B) Default non-planner mode ----------
        subtasks = self._decompose_task(task)
        if not subtasks:
            subtasks = [self._mk_subtask(task_text, ctx, i=1, requirement="general-reasoning")]

        # normalize
        for i, st in enumerate(subtasks):
            if not isinstance(st, dict):
                st = {"input": str(st)}
                subtasks[i] = st
            st.setdefault("id", f"sub_{i + 1}")
            st.setdefault("requirement", "general-reasoning")
            st.setdefault("context", ctx)
            st.setdefault("original_task", task_text)
            st.setdefault("description", st.get("input") or st.get("description") or task_text)
            if not st.get("input"):
                st["input"] = st.get("description") or task_text

            # normalize req for matching
            st["requirement"] = self._normalize_requirement(str(st.get("requirement", "general-reasoning")))

        agent_assignments = self._find_suitable_agents(subtasks)

        out = self._execute_with_cot_voting(
            subtasks=subtasks,
            agent_assignments=agent_assignments,
            cot_count=cot_count,
            return_mode=return_mode,
        )

        if return_mode == "trace":
            if isinstance(out, dict) and "results" in out and isinstance(out["results"], dict):
                if len(out["results"]) == 1:
                    one = next(iter(out["results"].values()))
                    out["final"] = self._extract_final_from_text(str(one)) or str(one)
            return out

        if return_mode == "final":
            if isinstance(out, dict) and len(out) == 1:
                s = str(next(iter(out.values()))).strip()
                s = re.sub(r"(?is)</?\s*answer\s*>", "", s).strip()
                fin = self._extract_final_from_text(s)
                return (fin or s).strip()

            aggregated = self._aggregate_results(out, task)  # type: ignore[arg-type]
            fin = self._extract_final_from_text(aggregated)
            return fin.strip() if fin else aggregated.strip()

        return self._aggregate_results(out, task)  # type: ignore[arg-type]

    # ---------------------- decomposition (simple baseline) ----------------------
    def _mk_subtask(self, task_text: str, ctx: Dict[str, Any], i: int, requirement: str) -> Dict[str, Any]:
        """
        ✅ P0-2: Create subtask dict with benchmark/difficulty_bin for priors lookup.
        
        These fields are required for routing.get_prior_success() to work correctly.
        """
        st = {
            "id": f"{uuid.uuid4().hex}_sub_{i}",
            "requirement": self._normalize_requirement(requirement),
            "input": task_text,
            "description": task_text,
            "context": ctx or {},
            "original_task": task_text,
        }
        # ✅ P0-2: Inject benchmark and difficulty_bin for priors lookup
        if isinstance(ctx, dict):
            st["benchmark"] = str(ctx.get("benchmark", "")).strip()
            st["difficulty_bin"] = str(ctx.get("difficulty_bin", ctx.get("difficulty", "unknown"))).strip() or "unknown"
        else:
            st["benchmark"] = ""
            st["difficulty_bin"] = "unknown"
        return st

    def _decompose_task(self, task: Task) -> List[Dict[str, Any]]:
        """
        Baseline decomposition: one subtask per requirement.
        """
        reqs = list(getattr(task, "requirements", []) or [])
        if not reqs:
            reqs = ["general-reasoning"]

        out: List[Dict[str, Any]] = []
        for i, r in enumerate(reqs, 1):
            out.append(
                self._mk_subtask(
                    task_text=getattr(task, "description", "") or "",
                    ctx=getattr(task, "context", {}) or {},
                    i=i,
                    requirement=str(r),
                )
            )
        return out

    # ---------------------- agent matching ----------------------
    def _agent_state(self, agent: Agent) -> Dict[str, Any]:
        """Symphony 2.0: read dynamic state if provided; else fallback defaults."""
        if hasattr(agent, "get_dynamic_state"):
            try:
                st = agent.get_dynamic_state()  # type: ignore[attr-defined]
                if isinstance(st, dict):
                    st.setdefault("available", True)
                    st.setdefault("load", 0.0)
                    st.setdefault("latency_ms", 500.0)
                    st.setdefault("reputation", 0.5)
                    return st
            except Exception:
                pass
        return {"available": True, "load": 0.0, "latency_ms": 500.0, "reputation": 0.5}

    def _find_suitable_agents(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        ✅ Return candidates per subtask with composite TopL score.
        
        Delegates to routing.select_topL() for clean separation of concerns.
        
        Returns:
          [{"agent": Agent, "match_score": float, "sim_emb": float, "prior_success": float}, ...]
        """
        try:
            from core.routing import select_topL
            use_routing = True
        except ImportError:
            use_routing = False
        
        assignments: Dict[str, List[Dict[str, Any]]] = {}
        for st in subtasks:
            sid = st["id"]
            
            # ✅ P0-3: Use routing module for clean TopL selection (must use routing)
            if use_routing:
                try:
                    candidates = select_topL(
                        agents=self.agents,
                        subtask=st,
                        topL=self.topL,
                        latency_scale_ms=float(self.latency_scale_ms),
                        use_embedding=True,  # ✅ P0-3: Must use embedding
                    )
                    # ✅ P0-7: Validate candidate schema
                    for cand in candidates:
                        required_keys = {"agent", "match_score", "sim_emb", "prior_success"}
                        missing = required_keys - set(cand.keys())
                        if missing:
                            raise ValueError(
                                f"P0-7: Candidate missing required keys: {missing}. "
                                f"Candidate keys: {list(cand.keys())}"
                            )
                    assignments[sid] = candidates
                    continue
                except Exception as e:
                    # ✅ P0-3: Strict routing mode: raise if routing fails (experiment mode)
                    if self.strict_routing:
                        raise RuntimeError(
                            f"P0-3: Routing.select_topL failed for subtask {sid} and strict_routing=True. "
                            f"Error: {e}"
                        )
                    # ✅ P0-3: Log routing failure but fallback to legacy (backward compatibility)
                    if self.verbose:
                        self._log(f"[WARN] Routing.select_topL failed for subtask {sid}: {e}. Falling back to legacy.")
                    # Fallback to legacy (for backward compatibility)
                    pass
            
            # Legacy fallback: capability_manager.match() only (should not happen in normal flow)
            req = self._normalize_requirement(str(st.get("requirement", "general-reasoning")))
            cand: List[Dict[str, Any]] = []
            for ag in self.agents:
                ms = 0.5
                if hasattr(ag, "capability_manager"):
                    try:
                        ms = float(ag.capability_manager.match(req))  # type: ignore[attr-defined]
                    except Exception:
                        ms = 0.5
                cand.append({
                    "agent": ag,
                    "match_score": ms,
                    "sim_emb": ms,  # Fallback: use match_score as sim_emb
                    "prior_success": 0.5,  # Fallback: default
                })
            cand.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)
            assignments[sid] = cand
        
        return assignments

    def _select_agent_dynamic(
            self,
            candidates: List[Dict[str, Any]],
            used_ids: set,
    ) -> Tuple[Agent, List[float], Dict[str, Any], float]:
        """
        Stage-1: Top-L by match_score (already sorted)
        Stage-2: Global LinUCB selects within Top-L using build_x(match, dynamic_state, available)

        Return: (agent, x, state, raw_match_score)
        """
        topL = candidates[: self.topL]

        pool: List[Tuple[str, List[float], Agent, Dict[str, Any], float]] = []
        for c in topL:
            agent = c["agent"]
            # ✅ P0-1: Use unified agent key resolution (not just agent_id)
            aid = self._resolve_agent_key(agent)
            if not aid:
                # Fallback: use object id if all keys are empty (should not happen)
                aid = f"agent_{id(agent)}"

            if aid in used_ids and len(used_ids) < len(topL):
                continue

            st = self._agent_state(agent)
            if not bool(st.get("available", True)):
                continue

            # ✅ P1: UCB stage: use build_x_from_candidate() to reuse TopL results (avoid recompute)
            try:
                from core.routing import build_x_from_candidate
                x, sim_emb_val, prior_success_val = build_x_from_candidate(
                    candidate=c,
                    agent=agent,
                    latency_scale_ms=float(self.latency_scale_ms),
                )
                raw_ms = float(c.get("match_score", 0.0))  # TopL composite score (for logging)
            except Exception:
                # ✅ P0-4: Fallback: use unified helper (ensures ms=sim_emb, rep=prior_success)
                x = self._build_x_from_candidate_or_fallback(
                    candidate=c,
                    agent=agent,
                    dynamic_state=st,
                )
                raw_ms = float(c.get("match_score", 0.0))
            pool.append((aid, x, agent, st, raw_ms))

        # ✅ safer fallback: pick first AVAILABLE in topL; else pick topL[0] but mark available=False in x
        if not pool:
            for c in topL:
                ag = c["agent"]
                st0 = self._agent_state(ag)
                if bool(st0.get("available", True)):
                    # ✅ P0-4: Use unified helper (ensures ms=sim_emb, rep=prior_success)
                    x0 = self._build_x_from_candidate_or_fallback(
                        candidate=c,
                        agent=ag,
                        dynamic_state=st0,
                    )
                    raw_ms0 = float(c.get("match_score", 0.0))  # For logging
                    return ag, x0, st0, raw_ms0

            c0 = topL[0]
            ag0 = c0["agent"]
            st0 = self._agent_state(ag0)
            # ✅ P0-4: Use unified helper (ensures ms=sim_emb, rep=prior_success)
            x0 = self._build_x_from_candidate_or_fallback(
                candidate=c0,
                agent=ag0,
                dynamic_state=st0,
            )
            raw_ms0 = float(c0.get("match_score", 0.0))  # For logging
            return ag0, x0, st0, raw_ms0

        assert self.selector is not None
        chosen_id = self.selector.select([(aid, x) for (aid, x, _, _, _) in pool])
        for (aid, x, agent, st, raw_ms) in pool:
            if aid == chosen_id:
                return agent, x, st, raw_ms
        return pool[0][2], pool[0][1], pool[0][3], pool[0][4]

    # ---------------------- core: multi-CoT + voting (+ trace) ----------------------
    def _execute_with_cot_voting(
            self,
            subtasks: List[Dict[str, Any]],
            agent_assignments: Dict[str, List[Dict[str, Any]]],
            cot_count: int,
            return_mode: str = "final",  # "final" | "aggregate" | "trace"
    ) -> Any:
        """
        For each subtask:
        - run up to `cot_count` times (bounded by |TopL|)
        - vote among outputs (keys on extracted Final)
        - update LinUCB online after vote (winner bonus + latency penalty + optional correctness reward)
        """
        results: Dict[str, str] = {}
        traces_by_subtask: Dict[str, Any] = {}

        for st in subtasks:
            sid = st["id"]
            # 改动4: 默认requirement改为math_reasoning（如果context中有benchmark=gsm8k）
            ctx_check = st.get("context", {}) or {}
            bench_check = str(ctx_check.get("benchmark", "")).strip().lower()
            default_req = "math_reasoning" if bench_check in {"gsm8k", "gsm"} else "general-reasoning"
            req = str(st.get("requirement", default_req))
            candidates = agent_assignments.get(sid, [])

            if not candidates:
                err = f"[ERROR] No agents available for subtask: {req}"
                results[sid] = err
                if return_mode == "trace":
                    traces_by_subtask[sid] = {"error": err, "runs": [], "voted": err, "requirement": req}
                continue

            # ✅ Cold_start round-robin mode: check if context has _cold_start_task_index
            ctx = st.get("context", {}) or {}
            cold_start_task_index = ctx.get("_cold_start_task_index")
            cold_start_agent_keys = ctx.get("_cold_start_agents", [])
            
            if cold_start_task_index is not None and cold_start_agent_keys:
                # ✅ Cold_start round-robin: each task -> exactly one agent (round-robin)
                # Select agent by round-robin: task_index % len(agents)
                agent_key_idx = int(cold_start_task_index) % len(cold_start_agent_keys) if cold_start_agent_keys else 0
                target_agent_key = cold_start_agent_keys[agent_key_idx]
                
                # Find agent by key
                selected_agent = None
                selected_candidate = None
                for c in candidates:
                    ag = c["agent"]
                    aid = self._resolve_agent_key(ag)
                    if aid == target_agent_key:
                        selected_agent = ag
                        selected_candidate = c
                        break
                
                if selected_agent is None:
                    # ✅ Fallback: still use round-robin from available candidates
                    # Match agents in candidates by key, then pick by round-robin index
                    available_candidates = []
                    candidate_keys = []
                    for c in candidates:
                        ag = c["agent"]
                        stt = self._agent_state(ag)
                        if bool(stt.get("available", True)):
                            aid = self._resolve_agent_key(ag)
                            available_candidates.append((aid, ag, c))
                            candidate_keys.append(aid)
                    
                    if available_candidates:
                        # Find target_agent_key's position in sorted agent_keys list
                        # Use that position to select from available candidates
                        try:
                            target_idx_in_all = cold_start_agent_keys.index(target_agent_key)
                            # Find the first available candidate whose key matches any agent in sorted list at same relative position
                            # Simplified: use round-robin index directly on available candidates
                            fallback_idx = int(cold_start_task_index) % len(available_candidates) if available_candidates else 0
                            selected_agent = available_candidates[fallback_idx][1]
                            selected_candidate = available_candidates[fallback_idx][2]
                        except (ValueError, IndexError):
                            # If target not found or index error, use first available
                            selected_agent = available_candidates[0][1]
                            selected_candidate = available_candidates[0][2]
                
                if selected_agent is None:
                    err = f"[ERROR] No agent found for cold_start round-robin (key={target_agent_key})"
                    results[sid] = err
                    if return_mode == "trace":
                        traces_by_subtask[sid] = {"error": err, "runs": [], "voted": err, "requirement": req}
                    continue
                
                # Execute once with selected agent
                aid = self._resolve_agent_key(selected_agent)
                if not aid:
                    aid = f"agent_{id(selected_agent)}"
                stt = self._agent_state(selected_agent)
                match_score = float(selected_candidate.get("match_score", 0.0)) if selected_candidate else 0.0
                x = self._build_x_from_candidate_or_fallback(
                    candidate=selected_candidate or {"agent": selected_agent, "sim_emb": 0.5, "prior_success": 0.5},
                    agent=selected_agent,
                    dynamic_state=stt,
                )
                
                t0 = time.time()
                try:
                    if self.dispatch_mode == "shared_bb":
                        requester = self._get_requester_agent()
                        if requester is None:
                            text = "[ERROR] dispatch_mode=shared_bb but no requester"
                        else:
                            run_tag = str(ctx.get("_run_id", "run")) + f":{sid}:cold_start"
                            text = self._execute_subtask_via_shared_bb(requester, st, run_tag=run_tag)
                    else:
                        text = self._execute_subtask_on_agent(selected_agent, st)
                except Exception as e:
                    text = f"[AGENT_ERROR] {str(e)}"
                
                dt_ms = (time.time() - t0) * 1000.0
                final_result = text
                run_records = [{
                    "agent_id": aid,
                    "match_score": float(match_score),
                    "sim_emb": float(selected_candidate.get("sim_emb", 0.5)) if selected_candidate else 0.5,
                    "prior_success": float(selected_candidate.get("prior_success", 0.5)) if selected_candidate else 0.5,
                    "x": x,
                    "latency_ms": float(dt_ms),
                    "text": text,
                    "final": self._extract_final_from_text(text) or "",
                }]
                
                results[sid] = final_result
                if return_mode == "trace":
                    traces_by_subtask[sid] = {
                        "requirement": req,
                        "context": ctx,
                        "runs": run_records,
                        "voted": final_result,
                        "voted_final": self._extract_final_from_text(final_result) or "",
                    }
                continue

            # Normal mode: filter available and use Top-L + Multi-CoT
            # filter available
            candidates_avail: List[Dict[str, Any]] = []
            for c in candidates:
                ag = c["agent"]
                stt = self._agent_state(ag)
                if bool(stt.get("available", True)):
                    candidates_avail.append(c)

            if not candidates_avail:
                err = f"[ERROR] No AVAILABLE agents for subtask: {req}"
                results[sid] = err
                if return_mode == "trace":
                    traces_by_subtask[sid] = {"error": err, "runs": [], "voted": err, "requirement": req}
                continue

            # ✅ Exploration constraint 1: Top-L must be unique (deduplicate by agent_id)
            # Ensure no duplicate agents in topL (critical for exploration)
            seen_agent_ids = set()
            topL_unique: List[Dict[str, Any]] = []
            for c in candidates_avail:
                ag = c["agent"]
                aid = self._resolve_agent_key(ag)
                if not aid:
                    aid = f"agent_{id(ag)}"
                if aid not in seen_agent_ids:
                    seen_agent_ids.add(aid)
                    topL_unique.append(c)
                    if len(topL_unique) >= self.topL:
                        break
            
            # If not enough unique agents, use what we have (at least 1)
            topL = topL_unique[: max(1, self.topL)] if topL_unique else candidates_avail[:1]
            
            # ✅ Exploration constraint 2: Fixed exploration budget K (don't vary with topL length)
            # 改动5: 对于GSM8K，实现self-consistency（同agent多次采样）
            ctx_check = st.get("context", {}) or {}
            bench_check = str(ctx_check.get("benchmark", "")).strip().lower()
            is_gsm8k_self_consistency = bench_check in {"gsm8k", "gsm"} and cot_count >= 3
            
            runs = int(cot_count) if cot_count > 0 else 0
            if runs <= 0:
                err = f"[ERROR] All agents filtered out for subtask: {req}"
                results[sid] = err
                if return_mode == "trace":
                    traces_by_subtask[sid] = {"error": err, "runs": [], "voted": err, "requirement": req}
                continue

            used_ids = set()
            run_records: List[Dict[str, Any]] = []
            cot_results: List[str] = []
            
            # 改动5: self-consistency模式：选择第一个agent，然后多次采样
            selected_agent = None
            selected_candidate = None
            selected_x = None
            selected_stt = None
            
            if is_gsm8k_self_consistency:
                # 选择第一个agent（通过UCB或round-robin）
                if self.use_dynamic and self.selector is not None:
                    selected_agent, selected_x, selected_stt, _ = self._select_agent_dynamic(topL, used_ids)
                    # 找到对应的candidate
                    for c in topL:
                        if c["agent"] == selected_agent:
                            selected_candidate = c
                            break
                else:
                    if topL:
                        selected_candidate = topL[0]
                        selected_agent = selected_candidate["agent"]
                        selected_stt = self._agent_state(selected_agent)
                        selected_x = self._build_x_from_candidate_or_fallback(
                            candidate=selected_candidate,
                            agent=selected_agent,
                            dynamic_state=selected_stt,
                        )

            for j in range(runs):
                if is_gsm8k_self_consistency and selected_agent:
                    # Self-consistency: 使用同一个agent多次采样
                    agent = selected_agent
                    x = selected_x
                    stt = selected_stt
                    match_score = float(selected_candidate.get("match_score", 0.0)) if selected_candidate else 0.0
                elif self.use_dynamic and self.selector is not None:
                    agent, x, _st, match_score = self._select_agent_dynamic(topL, used_ids)
                else:
                    # ✅ cold_start: static Top-L but round-robin across agents (no repeat)
                    # Use j % len(topL) to cycle through topL candidates
                    candidate_idx = j % len(topL) if topL else 0
                    candidate = topL[candidate_idx]
                    agent = candidate["agent"]
                    stt = self._agent_state(agent)
                    # ✅ P0-4: Use unified helper (ensures ms=sim_emb, rep=prior_success)
                    match_score = float(candidate.get("match_score", 0.0))  # For logging
                    x = self._build_x_from_candidate_or_fallback(
                        candidate=candidate,
                        agent=agent,
                        dynamic_state=stt,
                    )

                # ✅ P0-1: Use unified agent key resolution
                aid = self._resolve_agent_key(agent)
                if not aid:
                    aid = f"agent_{id(agent)}"
                if not is_gsm8k_self_consistency:
                    used_ids.add(aid)  # self-consistency模式下允许重复使用同一个agent


                # 改动4: 零容忍解析 + invalid时自动重试一次
                t0 = time.time()
                text = ""
                retry_count = 0
                max_retries = 1
                
                while retry_count <= max_retries:
                    try:
                        if self.dispatch_mode == "shared_bb":
                            requester = self._get_requester_agent()
                            if requester is None:
                                text = "[ERROR] dispatch_mode=shared_bb but no requester (agent with isep_client) registered"
                            else:
                                run_tag = str((st.get("context", {}) or {}).get("_run_id", "run")) + f":{st.get('id','sub')}:cot{len(run_records)+1}"
                                text = self._execute_subtask_via_shared_bb(requester, st, run_tag=run_tag)
                        else:
                            text = self._execute_subtask_on_agent(agent, st)
                    except Exception as e:
                        text = f"[AGENT_ERROR] {str(e)}"
                    
                    # 对于GSM8K，检查输出是否valid
                    ctx_check = st.get("context", {}) or {}
                    bench_check = str(ctx_check.get("benchmark", "")).strip().lower()
                    if bench_check in {"gsm8k", "gsm"} and text and not text.startswith("[ERROR]") and not text.startswith("[AGENT_ERROR]"):
                        parsed, is_valid, err = self._parse_strict_json(text, benchmark=bench_check)
                        if parsed is None or not is_valid:
                            # invalid输出，重试一次（低温度）
                            if retry_count < max_retries:
                                retry_count += 1
                                # 修改subtask的instruction，强调格式要求
                                original_instruction = st.get("input") or st.get("description") or ""
                                retry_instruction = (
                                    original_instruction
                                    + "\n\n[CRITICAL: RETRY AFTER INVALID OUTPUT]\n"
                                    + "Your previous output was INVALID. Output ONLY one JSON object.\n"
                                    + "NO extra text. NO multiple JSON objects.\n"
                                    + 'Format: {"final_answer": "<integer>", "valid": 1, "confidence": <0.0-1.0>}\n'
                                )
                                st_retry = st.copy()
                                st_retry["input"] = retry_instruction
                                st_retry["description"] = retry_instruction
                                st = st_retry
                                continue
                            else:
                                # 重试失败，标记为invalid
                                text = f"[INVALID_FORMAT] {err}: {text[:100]}"
                    
                    break  # 成功或达到最大重试次数

                dt_ms = (time.time() - t0) * 1000.0

                cot_results.append(text)
                # ✅ P1-1: Extract sim_emb and prior_success from candidate for trace (debug)
                cand_info = topL[0] if topL else {}
                run_records.append(
                    {
                        "agent_id": aid,
                        "match_score": float(match_score),  # TopL composite score
                        "sim_emb": float(cand_info.get("sim_emb", 0.5)),  # ✅ P1-1: For debug
                        "prior_success": float(cand_info.get("prior_success", 0.5)),  # ✅ P1-1: For debug
                        "x": x,
                        "latency_ms": float(dt_ms),
                        "text": text,
                        "final": self._extract_final_from_text(text) or "",
                    }
                )

            final_result = self._vote_on_results(cot_results, st)

            # optional risk guard
            if self.enable_risk_guard and self.risk_guard is not None:
                decision = self.risk_guard.assess(
                    task_context=st.get("context", {}) or {},
                    subtask=st,
                    cot_results=cot_results,
                    voted_text=final_result,
                    history_risk=0.0,
                )
                if getattr(decision, "action", None) == "rerun":
                    extra = int(getattr(getattr(self.risk_guard, "cfg", None), "extra_cot", 1))
                    for _ in range(extra):
                        if self.use_dynamic and self.selector is not None:
                            agent, x, _st, match_score = self._select_agent_dynamic(topL, used_ids)
                        else:
                            # ✅ P0-2: Use unified helper for consistent feature definitions
                            agent = topL[0]["agent"]
                            stt = self._agent_state(agent)
                            match_score = float(topL[0].get("match_score", 0.0))
                            x = self._build_x_from_candidate_or_fallback(
                                candidate=topL[0],
                                agent=agent,
                                dynamic_state=stt,
                            )

                        # ✅ P0-A: Use unified agent key resolution (not just agent_id)
                        aid = self._resolve_agent_key(agent)
                        if not aid:
                            aid = f"agent_{id(agent)}"
                        used_ids.add(aid)
                        t0 = time.time()
                        try:
                            text = self._execute_subtask_on_agent(agent, st)
                        except Exception as e:
                            text = f"[AGENT_ERROR] {str(e)}"
                        dt_ms = (time.time() - t0) * 1000.0

                        cot_results.append(text)
                        run_records.append(
                            {
                                "agent_id": aid,
                                "match_score": float(match_score),
                                "x": x,
                                "latency_ms": float(dt_ms),
                                "text": text,
                                "final": self._extract_final_from_text(text) or "",
                            }
                        )
                    final_result = self._vote_on_results(cot_results, st)

            results[sid] = final_result

            if self.dispatch_mode != "shared_bb":
                if self.use_dynamic and self.selector is not None and not self.eval_only:
                    self._online_update_after_vote(run_records, final_result, st)


            finals = []
            for r in run_records:
                f = (r.get("final") or "").strip()
                finals.append(f)

            vote_count = Counter(finals)

            vote_weight = defaultdict(float)
            for r in run_records:
                f = (r.get("final") or "").strip()
                vote_weight[f] += float(r.get("match_score", 0.0))  # 用 match_score 做一个“加权票”观测

            if return_mode == "trace":
                ctx = st.get("context", {}) or {}
                gold = self._get_gold_from_context(ctx)
                correct = self._is_correct(final_result, gold)
                traces_by_subtask[sid] = {
                    "requirement": req,
                    "context": ctx,
                    "gold": gold,
                    "vote_count": dict(vote_count),
                    "vote_weight_by_match_score": dict(vote_weight),
                    "correct": correct,
                    "runs": run_records,
                    "voted": final_result,
                    "voted_final": self._extract_final_from_text(final_result) or "",
                }

        if return_mode == "trace":
            return {"results": results, "traces": traces_by_subtask}
        return results

    def _online_update_after_vote(self, run_records, voted_text, subtask) -> None:
        """
        改动8: UCB update只用"最终被采纳的答案"并且必须valid
        - 先得到最终voted_final（且valid）
        - 只用这个reward更新一次
        - invalid/parse_fail：reward=0，但要单独记为format_error
        """
        ctx = subtask.get("context", {}) or {}
        benchmark = str(ctx.get("benchmark", "")).strip().lower()
        is_gsm8k = benchmark in {"gsm8k", "gsm"}
        gold = self._get_gold_from_context(ctx)
        
        # 严格验证voted_text
        parsed_voted, is_voted_valid, error_msg = self._parse_strict_json(voted_text, benchmark=benchmark if is_gsm8k else None)
        
        if parsed_voted is None or not is_voted_valid:
            # invalid的voted结果，不更新UCB（但记录format_error）
            for rec in run_records:
                rec["vote_format_error"] = error_msg
                rec["vote_reward"] = 0.0
            return
        
        voted_final = str(parsed_voted.get("final_answer", "")).strip()
        correct = self._is_correct(voted_final, gold) if gold else None
        
        # 改动8: 只用最终被采纳的答案更新一次
        # 找到对应的run（即产生voted_final的run）
        winner_rec = None
        for rec in run_records:
            x = rec.get("x")
            if not isinstance(x, list):
                continue

            latency_ms = float(rec.get("latency_ms", 0.0))
            text = str(rec.get("text", ""))

            ok = (text.strip() != "") and (not text.startswith("[ERROR]")) and (not text.startswith("[AGENT_ERROR]"))
            agent_final = self._extract_final_from_text(text) or text.strip()
            is_winner = (agent_final == voted_final) and (voted_final != "")

            lat_norm = min(1.0, latency_ms / max(1.0, self.latency_scale_ms))
            penalty = self.latency_penalty * (lat_norm ** 0.5)

            corr_term = 0.0
            if correct is True:
                corr_term += float(self.correctness_bonus)
            elif correct is False:
                corr_term -= float(self.incorrect_penalty)

            win_term = (self.win_bonus if is_winner else 0.0)
            base_term = (1.0 if ok else 0.0)

            reward = base_term + win_term + corr_term - penalty

            # ✅ 关键：把“每轮得分”写回 trace
            rec["vote_voted_final"] = voted_final
            rec["vote_ok"] = ok
            rec["vote_is_winner"] = is_winner
            rec["vote_base_term"] = base_term
            rec["vote_win_term"] = win_term
            rec["vote_corr_term"] = corr_term
            rec["vote_latency_penalty"] = penalty
            rec["vote_reward"] = reward

            self.selector.update(x, reward)
    # ---------------------- shared-bb dispatch helpers ----------------------
    def _get_requester_agent(self) -> Optional[Agent]:
        """
        Pick a requester agent that can broadcast beacons & delegate tasks.
        Requirement: agent.isep_client is not None.
        """
        # ✅ P1-2: explicit requester_id first (use unified key resolution)
        if self.requester_id:
            for a in self.agents:
                agent_key = self._resolve_agent_key(a)
                if agent_key == str(self.requester_id) and getattr(a, "isep_client", None) is not None:
                    return a

        # otherwise: first agent with isep_client
        for a in self.agents:
            if getattr(a, "isep_client", None) is not None:
                return a
        return None

    def _execute_subtask_via_shared_bb(self, requester: Agent, subtask: Dict[str, Any], run_tag: str) -> str:
        """
        Synchronous subtask execution via:
          requester.assign_task() -> (beacon + LinUCB select + by-ref delegate)
          then poll blackboard KV for result:{executor}.
        """
        if getattr(requester, "isep_client", None) is None:
            return "[ERROR] requester has no isep_client"

        bb = getattr(requester, "bb", None)
        if bb is None:
            return "[ERROR] requester has no blackboard instance"

        instruction = (
            subtask.get("input")
            or subtask.get("description")
            or subtask.get("original_task")
            or ""
        )
        requirement = str(subtask.get("requirement", "general-reasoning"))
        original_problem = subtask.get("original_task", instruction)
        ctx = dict(subtask.get("context", {}) or {})

        # ✅ unique base tid per run (avoid KV collision across subtasks/cot runs)
        base_tid = f"{run_tag}:{uuid.uuid4().hex[:8]}"

        # 从subtask id中提取子任务索引（如 "xxx_sub_2" -> 2）
        subtask_id_str = str(subtask.get("id", ""))
        subtask_index = 1
        if "_sub_" in subtask_id_str:
            try:
                parts = subtask_id_str.split("_sub_")
                if len(parts) > 1:
                    subtask_index = int(parts[-1])
            except (ValueError, IndexError):
                subtask_index = 1
        
        # build a minimal task dict compatible with Agent.assign_task/execute_task
        agent_task = {
            "task_id": base_tid,     # will be overwritten to dispatch_id inside assign_task (task_key)
            "subtask_id": subtask_index,  # 使用实际的子任务索引
            "steps": {str(subtask_index): [instruction, requirement]},
            "previous_results": subtask.get("previous_results", []) or [],
            "original_problem": original_problem,
            "final_result": "",
            "context": ctx,          # requester.assign_task will inject dispatch_id here
            "user_id": "symphony_orchestrator",
        }

        # delegate (by-ref if requester.use_shared_bb=True)
        requester.assign_task(agent_task, topL=self.topL)

        # after assign_task, task_id has been enforced to dispatch_id (task_key)
        dispatch_id = str(agent_task.get("task_id", ""))
        if not dispatch_id:
            return "[ERROR] dispatch_id empty after assign_task"

        # pending record exists on requester
        pending = getattr(requester, "_pending", {}).get(dispatch_id, None)
        if not pending:
            # could happen if something popped it unexpectedly
            return "[ERROR] requester pending not found (dispatch_id mismatch)"

        executor = str(pending.get("executor", ""))
        if not executor:
            return "[ERROR] executor empty in pending"

        # wait result in BB kv
        t0 = time.time()
        while True:
            res = bb.kv_get(dispatch_id, f"result:{executor}", default=None)
            if res:
                # normalize
                if isinstance(res, dict):
                    out_text = str(res.get("result", "")).strip()
                    # ✅ update requester LinUCB (and pop pending) deterministically
                    try:
                        res.setdefault("task_id", dispatch_id)
                        requester.on_task_result(res)
                    except Exception:
                        pass
                    return out_text
                else:
                    out_text = str(res).strip()
                    try:
                        requester.on_task_result({"task_id": dispatch_id, "result": out_text})
                    except Exception:
                        pass
                    return out_text

            if time.time() - t0 > self.shared_timeout_s:
                return f"[ERROR] Timeout waiting result in blackboard (dispatch_id={dispatch_id}, executor={executor})"

            time.sleep(self.shared_poll_interval)

    # ---------------------- calling an agent safely ----------------------
    def _execute_subtask_on_agent(self, agent: Agent, subtask: Dict[str, Any]) -> str:
        """
        Compatibility shim:
        build legacy agent_task and call agent.execute_task(...).
        """
        instruction = (
                subtask.get("input")
                or subtask.get("description")
                or subtask.get("original_task")
                or ""
        )
        requirement = str(subtask.get("requirement", "general-reasoning"))
        original_problem = subtask.get("original_task", instruction)
        
        # 从subtask id中提取子任务索引（如 "xxx_sub_2" -> 2）
        subtask_id_str = str(subtask.get("id", ""))
        subtask_index = 1
        if "_sub_" in subtask_id_str:
            try:
                parts = subtask_id_str.split("_sub_")
                if len(parts) > 1:
                    subtask_index = int(parts[-1])
            except (ValueError, IndexError):
                subtask_index = 1

        agent_task = {
            "subtask_id": subtask_index,  # 使用实际的子任务索引，而不是硬编码1
            "steps": {str(subtask_index): [instruction, requirement]},
            "previous_results": subtask.get("previous_results", []) or [],
            "original_problem": original_problem,
            "final_result": "",
            "user_id": "symphony_orchestrator",
        }

        # no-model simulation path
        if hasattr(agent, "base_model") and getattr(agent, "base_model") is None:
            domain = (subtask.get("context", {}) or {}).get("domain", "general")
            return f"[SIMULATED] Agent {getattr(agent, 'agent_id', '')} handled {requirement} ({domain})."

        if not hasattr(agent, "execute_task"):
            return f"[ERROR] Agent {getattr(agent, 'agent_id', '')} has no execute_task()"

        result = agent.execute_task(agent_task)  # type: ignore[call-arg]

        # --- unwrap common return formats ---
        if result is None:
            return ""

        # avoid numeric/status-code becoming "1"
        if isinstance(result, (int, float, bool)):
            return ""

        if isinstance(result, str):
            return result.strip()

        if isinstance(result, dict):
            # OpenAI / OpenAI-compatible
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

            for k in ("final_result", "final_text", "text", "answer", "output", "content"):
                if k in result and result[k]:
                    return str(result[k]).strip()

            # Symphony might return {"<uuid>_sub_1": "<text>"}
            for v in result.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()

            return ""

        # object-like
        for attr in ("final_result", "final_text", "text", "answer", "output", "content"):
            if hasattr(result, attr):
                val = getattr(result, attr, None)
                if val:
                    return str(val).strip()

        if hasattr(result, "to_dict"):
            try:
                d = result.to_dict()
                if isinstance(d, dict):
                    for k in ("final_result", "final_text", "text", "answer", "output", "content"):
                        if k in d and d[k]:
                            return str(d[k]).strip()
            except Exception:
                pass

        return str(result).strip()

    # ---------------------- voting (改动3: 数值一致性优先 + 改动5: self-consistency) ----------------------
    def _vote_on_results(self, cot_results: List[str], subtask: Dict[str, Any]) -> str:
        if len(cot_results) == 1:
            return cot_results[0]

        # 获取benchmark信息用于验证
        ctx = subtask.get("context", {}) or {}
        benchmark = str(ctx.get("benchmark", "")).strip().lower()
        is_gsm8k = benchmark in {"gsm8k", "gsm"}

        # 改动2: 严格解析和验证每个run
        valid_runs: List[Tuple[str, str, float]] = []  # (text, final_answer, confidence)
        invalid_runs: List[str] = []
        
        for r in cot_results:
            if r.startswith("[ERROR]") or r.startswith("[AGENT_ERROR]"):
                invalid_runs.append(r)
                continue
            
            # 严格解析JSON
            parsed, is_valid, error_msg = self._parse_strict_json(r, benchmark=benchmark if is_gsm8k else None)
            
            if parsed is None or not is_valid:
                invalid_runs.append(r)
                continue
            
            final_answer = str(parsed.get("final_answer", "")).strip()
            confidence = float(parsed.get("confidence", 0.0)) if "confidence" in parsed else 0.0
            
            if final_answer:
                valid_runs.append((r, final_answer, confidence))
        
        # 如果没有valid的run，fallback到原来的逻辑
        if not valid_runs:
            valid_results = [r for r in cot_results if not r.startswith("[ERROR]") and not r.startswith("[AGENT_ERROR]")]
            return max(valid_results, key=len) if valid_results else cot_results[0]
        
        # 改动3: 数值一致性优先的投票策略（改进版：强制整数解析、自一致性优先）
        # 对于GSM8K，强制要求整数解析
        if is_gsm8k:
            # 过滤：只保留能解析为整数的答案
            integer_valid_runs: List[Tuple[str, str, float]] = []
            for text, final_answer, confidence in valid_runs:
                try:
                    # 规范化：去除引号、逗号、美元符号等
                    cleaned = final_answer.replace(",", "").replace("$", "").replace("%", "").strip()
                    # 去除外层引号
                    while len(cleaned) >= 2 and cleaned.startswith('"') and cleaned.endswith('"'):
                        cleaned = cleaned[1:-1].strip()
                    # 尝试解析为整数
                    int_val = int(cleaned)
                    # 使用规范化后的整数字符串
                    normalized_answer = str(int_val)
                    integer_valid_runs.append((text, normalized_answer, confidence))
                except (ValueError, TypeError, OverflowError):
                    # 无法解析为整数，跳过
                    continue
            
            if integer_valid_runs:
                valid_runs = integer_valid_runs
            else:
                # 如果没有能解析为整数的，fallback到原始逻辑
                pass
        
        # 1. 从每个valid run抽取数值答案（已规范化）
        answer_counts: Dict[str, int] = {}
        answer_confidences: Dict[str, List[float]] = {}
        answer_texts: Dict[str, List[str]] = {}  # 存储每个答案对应的原始text列表
        
        for text, final_answer, confidence in valid_runs:
            answer_counts[final_answer] = answer_counts.get(final_answer, 0) + 1
            if final_answer not in answer_confidences:
                answer_confidences[final_answer] = []
            answer_confidences[final_answer].append(confidence)
            if final_answer not in answer_texts:
                answer_texts[final_answer] = []
            answer_texts[final_answer].append(text)
        
        # 2. 自一致性优先：如果有多数票（相同数值出现≥2次），选多数
        max_count = max(answer_counts.values()) if answer_counts else 0
        if max_count >= 2:
            # 有多数票，选择出现次数最多的（自一致性）
            majority_answers = [ans for ans, cnt in answer_counts.items() if cnt == max_count]
            if len(majority_answers) == 1:
                # 唯一多数，返回对应的原始text（取第一个）
                target_answer = majority_answers[0]
                return answer_texts[target_answer][0]
            else:
                # 平票：在多数答案中选择confidence最高的
                best_answer = None
                best_confidence = -1.0
                for ans in majority_answers:
                    avg_conf = sum(answer_confidences[ans]) / len(answer_confidences[ans])
                    if avg_conf > best_confidence:
                        best_confidence = avg_conf
                        best_answer = ans
                if best_answer:
                    return answer_texts[best_answer][0]
        
        # 3. 如果没有多数票，选"最高confidence的数值"（但仅在自一致性之后）
        if answer_confidences:
            best_answer = None
            best_confidence = -1.0
            for ans, confs in answer_confidences.items():
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_answer = ans
            if best_answer:
                return answer_texts[best_answer][0]
        
        # 4. 如果confidence缺失，选"最常见数值"
        if answer_counts:
            most_common = max(answer_counts.items(), key=lambda kv: kv[1])[0]
            return answer_texts[most_common][0]
        
        # Fallback: 返回第一个valid run
        return valid_runs[0][0] if valid_runs else cot_results[0]

    def _weighted_vote(self, answers: List[str], weights: List[float]) -> str:
        if not answers:
            return ""
        if len(answers) == 1:
            return answers[0]
        score: Dict[str, float] = {}
        for a, w in zip(answers, weights):
            key = (a or "").strip()
            score[key] = score.get(key, 0.0) + float(w)
        return max(score.items(), key=lambda kv: kv[1])[0] if score else answers[0]

    # ---------------------- strict JSON parser with validation (改动2) ----------------------
    @staticmethod
    def _parse_strict_json(text: str, benchmark: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], bool, str]:
        """
        严格解析JSON输出，必须符合格式：{"final_answer":"<string>","valid":0/1}（confidence可选）
        
        返回: (parsed_dict, is_valid, error_msg)
        - parsed_dict: 解析出的JSON对象，如果解析失败则为None
        - is_valid: True表示格式正确且valid=1，False表示格式错误或valid=0
        - error_msg: 错误信息（如果is_valid=False）
        """
        if not isinstance(text, str) or not text.strip():
            return None, False, "empty_text"
        
        s_clean = text.strip()
        
        # 移除可能的```json ... ```包装
        if s_clean.startswith("```"):
            lines = s_clean.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s_clean = "\n".join(lines).strip()
        
        # 检查是否有多个JSON对象（禁止）
        json_count = s_clean.count("{")
        if json_count > 1:
            # 尝试找到第一个完整的JSON对象
            try:
                decoder = json.JSONDecoder()
                start = s_clean.find("{")
                if start >= 0:
                    obj, end_pos = decoder.raw_decode(s_clean[start:])
                    # 检查后面是否还有JSON
                    remaining = s_clean[start + end_pos:].strip()
                    if remaining and remaining.startswith("{"):
                        return None, False, "multiple_json"
            except Exception:
                return None, False, "multiple_json"
        
        # 检查是否有额外文本（禁止）
        start_idx = s_clean.find("{")
        end_idx = s_clean.rfind("}")
        if start_idx > 0 or (end_idx >= 0 and end_idx < len(s_clean) - 1):
            # 有文本在JSON前后
            if start_idx > 0:
                before = s_clean[:start_idx].strip()
                if before:
                    return None, False, "extra_text_before"
            if end_idx >= 0 and end_idx < len(s_clean) - 1:
                after = s_clean[end_idx + 1:].strip()
                if after:
                    return None, False, "extra_text_after"
        
        # 尝试解析JSON
        try:
            # 先尝试直接解析
            if s_clean.startswith("{") and s_clean.endswith("}"):
                data = json.loads(s_clean)
            else:
                # 尝试找到第一个JSON对象
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = s_clean[start_idx:end_idx + 1]
                    data = json.loads(json_str)
                else:
                    return None, False, "no_json"
            
            if not isinstance(data, dict):
                return None, False, "not_dict"
            
            # 检查必需字段
            if "final_answer" not in data:
                return None, False, "missing_final_answer"
            
            if "valid" not in data:
                return None, False, "missing_valid"
            
            final_answer_raw = str(data.get("final_answer", "")).strip()
            valid_flag = data.get("valid")
            
            # 验证valid字段
            if valid_flag not in (0, 1, "0", "1", True, False):
                return None, False, "invalid_valid_field"
            
            is_valid_flag = (valid_flag == 1 or valid_flag == "1" or valid_flag is True)
            
            # 规范化final_answer：处理转义引号（如 "\\"8\\"" -> "8"）
            final_answer = final_answer_raw
            # 重复去除外层引号，直到没有引号包裹
            while len(final_answer) >= 2 and final_answer.startswith('"') and final_answer.endswith('"'):
                try:
                    # 尝试JSON解析来去除转义引号
                    final_answer = json.loads(final_answer)
                    if isinstance(final_answer, str):
                        final_answer = final_answer.strip()
                    else:
                        final_answer = str(final_answer).strip()
                except (json.JSONDecodeError, TypeError):
                    # 如果JSON解析失败，手动去除外层引号
                    final_answer = final_answer[1:-1].strip()
                    # 如果去除引号后还是引号包裹，继续
                    if len(final_answer) >= 2 and final_answer.startswith('"') and final_answer.endswith('"'):
                        continue
                    break
            
            # 更新data中的final_answer为规范化后的值
            data["final_answer"] = final_answer
            
            # 对于GSM8K，验证final_answer必须是可解析的整数
            if benchmark and benchmark.lower() in {"gsm8k", "gsm"}:
                if not final_answer:
                    return data, False, "empty_final_answer"
                
                # 清理字符串：去除逗号、美元符号、百分号等
                cleaned = final_answer.replace(",", "").replace("$", "").replace("%", "").strip()
                
                # 检查是否匹配整数正则表达式（允许负号）
                import re
                if not re.match(r'^-?\d+$', cleaned):
                    return data, False, "not_integer_format"
                
                try:
                    # 尝试解析为整数
                    num_val = int(cleaned)
                    # 验证成功，更新data中的final_answer为规范化整数字符串
                    data["final_answer"] = str(num_val)
                except (ValueError, TypeError, OverflowError):
                    return data, False, "not_numeric"
            
            return data, is_valid_flag, ""
            
        except json.JSONDecodeError as e:
            return None, False, f"json_decode_error: {str(e)}"
        except Exception as e:
            return None, False, f"parse_error: {str(e)}"
    
    # ---------------------- robust final extractor (BBH-friendly) ----------------------
    @staticmethod
    def _extract_final_from_text(s: str) -> Optional[str]:
        if not isinstance(s, str):
            return None

        # 0) JSON format: {"final_answer": "...", "confidence": 0.9, "valid": 1, "abstain": 0}
        # ✅ Priority 1: Try to parse JSON and extract final_answer field
        try:
            import json
            # Try to extract JSON object from text
            s_clean = s.strip()
            if s_clean.startswith("{") and s_clean.endswith("}"):
                # Direct JSON string
                data = json.loads(s_clean)
                if isinstance(data, dict) and "final_answer" in data:
                    final_ans = str(data.get("final_answer", "")).strip()
                    if final_ans:
                        return final_ans
            else:
                # Try to find JSON object within text
                start_idx = s_clean.find("{")
                end_idx = s_clean.rfind("}")
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = s_clean[start_idx:end_idx + 1]
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "final_answer" in data:
                        final_ans = str(data.get("final_answer", "")).strip()
                        if final_ans:
                            return final_ans
        except (json.JSONDecodeError, ValueError, KeyError):
            # JSON parsing failed, continue with regex-based extraction
            pass
        except Exception:
            # Any other exception (e.g., json module not available), fall through
            pass

        # 1) boxed
        m = re.findall(r"\\boxed\{([^}]*)\}", s)
        if m:
            return m[-1].strip()

        # 2) explicit "Final answer:" / "答案："
        m2 = re.search(r"(?im)^\s*(?:final\s*answer|answer|答案)\s*[:：]\s*([^\n\r]+)\s*$", s)
        if m2:
            cand = m2.group(1).strip()

            # ✅ dyck: prioritize bracket-only substring BEFORE any split on '['
            cand_ns = re.sub(r"\s+", "", cand)
            brs = re.findall(r"[<>\[\]\(\)\{\}]+", cand_ns)
            if brs:
                return max(brs, key=len)

            # other tasks: trim trailing punctuation/paren
            cand = re.split(r"\s*(?:\.\s*|,|;|\(|（|【)", cand, maxsplit=1)[0].strip()
            if cand:
                return cand

        # 2.5) dyck: whole-line brackets
        t = s.strip()
        if re.fullmatch(r"[<>\[\]\(\)\{\}]+", t or ""):
            return t

        # 2.6) validity (formal_fallacies)
        m = re.search(r"(?i)\b(valid|invalid)\b", s)
        if m:
            return m.group(1).lower()

        # 3) True/False
        m = re.search(r"\b(true|false)\b", s, re.I)
        if m:
            return m.group(1).lower().capitalize()

        # 4) yes/no
        m = re.search(r"\b(yes|no)\b", s, re.I)
        if m:
            return m.group(1).lower()

        # 5) single choice A-Z
        m = re.search(r"\b([A-Za-z])\b(?![A-Za-z])", s)
        if m:
            return m.group(1).upper()

        # 6) numbers (last)
        m = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", s)
        if m:
            return m[-1]

        return None

    # ---------------------- aggregation ----------------------
    def _aggregate_results(self, results: Dict[str, str], original_task: Task) -> str:
        aggregated = "## Symphony Multi-Agent Task Execution Result\n\n"
        aggregated += f"**Original Task**: {getattr(original_task, 'description', '')}\n\n"
        aggregated += f"**Domain**: {getattr(original_task, 'context', {}).get('domain', 'General')}\n"
        aggregated += f"**Complexity**: {getattr(original_task, 'context', {}).get('complexity', 'Medium')}\n\n"
        aggregated += "### Subtask Results:\n\n"

        ctx = getattr(original_task, 'context', {}) or {}
        benchmark = str(ctx.get('benchmark', '')).strip().lower()
        is_gsm8k = benchmark in {"gsm8k", "gsm"}
        original_text = getattr(original_task, 'description', '') or ''

        finals: List[Tuple[str, str, int]] = []  # (answer, sid, subtask_index)
        for i, (sid, result) in enumerate(results.items(), 1):
            aggregated += f"{i}. **{sid}**: {result}\n\n"
            ext = self._extract_final_from_text(result)
            if ext:
                # 提取子任务索引（用于排序和选择最终答案）
                subtask_index = i
                if "_sub_" in sid:
                    try:
                        parts = sid.split("_sub_")
                        if len(parts) > 1:
                            subtask_index = int(parts[-1])
                    except (ValueError, IndexError):
                        subtask_index = i
                finals.append((ext, sid, subtask_index))

        aggregated += (
            f"\n**Execution Summary**: Coordinated {len(results)} subtasks "
            f"via Top-L + Global LinUCB selection and CoT voting.\n"
        )

        # 对于GSM8K，优先选择最后一个子任务的答案（最终合成步骤）
        # 并应用合理性检查
        if is_gsm8k and finals:
            # 按子任务索引排序
            finals_sorted = sorted(finals, key=lambda x: x[2])
            
            # 优先选择最后一个子任务的答案
            final_candidates = [f for f in finals_sorted if f[2] == finals_sorted[-1][2]]
            
            # 如果有多个候选，应用合理性检查
            if len(final_candidates) > 1:
                # 应用合理性检查：选择最合理的答案
                best_answer = self._apply_gsm8k_sanity_check(final_candidates, original_text)
                if best_answer:
                    aggregated += f"\n**Final answer**: {best_answer}\n"
                    return aggregated.strip()
            
            # 使用最后一个子任务的答案
            final_ans = final_candidates[0][0] if final_candidates else finals_sorted[-1][0]
        else:
            # 非GSM8K：使用最后一个提取的答案
            final_ans = finals[-1][0] if finals else None
        
        if final_ans:
            aggregated += f"\n**Final answer**: {final_ans}\n"
        return aggregated.strip()
    
    def _apply_gsm8k_sanity_check(self, candidates: List[Tuple[str, str, int]], original_text: str) -> Optional[str]:
        """
        对GSM8K候选答案应用合理性检查。
        
        返回最合理的答案，如果没有合理的则返回None。
        """
        if not candidates:
            return None
        
        # 提取候选答案的数值
        candidate_values: List[Tuple[int, str]] = []
        for answer, sid, _ in candidates:
            try:
                # 规范化答案
                cleaned = answer.replace(",", "").replace("$", "").replace("%", "").strip()
                # 去除引号
                while len(cleaned) >= 2 and cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1].strip()
                val = int(cleaned)
                candidate_values.append((val, answer))
            except (ValueError, TypeError, OverflowError):
                continue
        
        if not candidate_values:
            return candidates[0][0]  # 如果无法解析，返回第一个原始答案
        
        # 合理性检查1：如果问题提到"$"或"dollars"，答案应该是合理的货币值
        if "$" in original_text.lower() or "dollar" in original_text.lower():
            # 货币值通常不会太小（除非明确说明）
            # 优先选择较大的值（可能是最终总价而非单价）
            candidate_values.sort(key=lambda x: x[0], reverse=True)
            return candidate_values[0][1]
        
        # 合理性检查2：如果问题提到"days"或"minutes"，答案不应该过大
        if "day" in original_text.lower():
            # 天数通常不会超过1000（除非是特殊场景）
            reasonable = [v for v in candidate_values if v[0] <= 1000]
            if reasonable:
                return reasonable[0][1]
        
        if "minute" in original_text.lower():
            # 分钟数通常不会超过10000
            reasonable = [v for v in candidate_values if v[0] <= 10000]
            if reasonable:
                return reasonable[0][1]
        
        # 合理性检查3：如果问题提到"per"或"each"，答案可能是倍数关系
        # 选择最接近中位数的值
        if len(candidate_values) > 1:
            values = [v[0] for v in candidate_values]
            median = sorted(values)[len(values) // 2]
            # 选择最接近中位数的
            best = min(candidate_values, key=lambda x: abs(x[0] - median))
            return best[1]
        
        # 默认：返回第一个
        return candidate_values[0][1]

    # ---------------------- Symphony 1.0 planner (optional) ----------------------
    _PLANNER_PROMPT = """You are a problem decomposer, NOT a solver.
Break the problem into a sequence of executable subtasks.
Do NOT solve the problem.

Return STRICT JSON ONLY (no markdown), format:
{
  "subtasks": [
    "Q1: ...",
    "Q2: ...",
    ...
  ]
}

Rules:
- Decide the number of subtasks based on task complexity, but keep it between 3 and 8 inclusive.
- No final answer, no intermediate results.
- The LAST subtask must instruct producing the final answer in the required output format for the whole task.
Problem:
{user_input}
"""

    def _select_planning_agents(self, k: int) -> List[Agent]:
        planners = [a for a in self.agents if "planning" in (getattr(a, "capabilities", []) or [])]
        if len(planners) >= k:
            return planners[:k]
        return list(self.agents)[:k]

    def _parse_planner_json(self, raw: str) -> Optional[List[str]]:
        try:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            s = m.group(0) if m else raw
            j = json.loads(s)
            arr = j.get("subtasks", None)
            if isinstance(arr, list) and arr:
                return [str(x) for x in arr if str(x).strip()]
        except Exception:
            return None
        return None

    def _plan_chains_v1(self, task_text: str, ctx: Dict[str, Any], m: int) -> List[Dict[str, Any]]:
        planners = self._select_planning_agents(m)
        plans: List[Dict[str, Any]] = []

        def _normalize_subtasks(subtasks_txt: List[str]) -> List[str]:
            # Enforce 3-8 subtasks for planner mode
            cleaned = [s for s in (subtasks_txt or []) if str(s).strip()]
            if len(cleaned) > 8:
                cleaned = cleaned[:8]
            if len(cleaned) < 3:
                padding = [
                    "Identify key quantities and constraints in the problem.",
                    "Solve the problem step-by-step with clear reasoning.",
                    "Produce the final answer in the required output format for the whole task.",
                ]
                needed = 3 - len(cleaned)
                cleaned = cleaned + padding[:needed]
            # Ensure last step instructs final answer
            if cleaned:
                cleaned[-1] = "Produce the final answer in the required output format for the whole task."
            return cleaned

        for p in planners:
            tmpl = self._PLANNER_PROMPT
            tmpl = tmpl.replace("{", "{{").replace("}", "}}").replace("{{user_input}}", "{user_input}")
            prompt = tmpl.format(user_input=task_text)

            fake_subtask = {
                "id": f"plan_{uuid.uuid4().hex}",
                "requirement": "planning",
                "input": prompt,
                "description": prompt,
                "original_task": task_text,
                "context": ctx or {},
            }
            raw = self._execute_subtask_on_agent(p, fake_subtask)
            subtasks_txt = self._parse_planner_json(raw)

            if not subtasks_txt:
                continue

            chain: List[Dict[str, Any]] = []
            for i, q in enumerate(_normalize_subtasks(subtasks_txt), 1):
                # ✅ P0-B: Use _mk_subtask() to ensure benchmark/difficulty_bin injection
                st_base = self._mk_subtask(task_text, ctx or {}, i, "general-reasoning")
                # Override input/description with planner's subtask text
                st_base["input"] = q
                st_base["description"] = q
                chain.append(st_base)
            if chain:
                plans.append({"planner": getattr(p, "agent_id", ""), "chain": chain})

        if not plans:
            # Fallback: still respect 3-8 subtasks requirement
            fallback_subtasks = _normalize_subtasks([])
            chain: List[Dict[str, Any]] = []
            for i, q in enumerate(fallback_subtasks, 1):
                st_base = self._mk_subtask(task_text, ctx, i=i, requirement="general-reasoning")
                st_base["input"] = q
                st_base["description"] = q
                chain.append(st_base)
            plans = [{"planner": "fallback", "chain": chain}]
        return plans

    def _format_executor_input(self, base_task: str, q: str, prev: List[str]) -> str:
        s = [base_task.strip()]
        if prev:
            s.append("Previous results:\n" + "\n".join(prev))
        s.append(f"Current sub-task: {q}")
        return "\n\n".join(s).strip()

    def _execute_one_subtask_beacon(self, st: Dict[str, Any], used_ids: set, base_ctx: Dict[str, Any], step_index: int = 0) -> Dict[
        str, Any]:
        """
        Planner step execution:
        - Top-L + LinUCB selection
        - Execute once per step
        - ✅ base online update here too (closes loop in planner mode)
        """
        assigns = self._find_suitable_agents([st])
        cands = assigns.get(st["id"], []) or []
        if not cands:
            return {"text": "[ERROR] No agents", "match_score": 0.0, "x": None, "latency_ms": 0.0, "agent_id": None,
                    "final": ""}

        # ✅ Exploration constraint 1: Top-L must be unique (deduplicate by agent_id)
        seen_agent_ids = set()
        topL_unique: List[Dict[str, Any]] = []
        for c in cands:
            ag = c["agent"]
            aid = self._resolve_agent_key(ag)
            if not aid:
                aid = f"agent_{id(ag)}"
            if aid not in seen_agent_ids:
                seen_agent_ids.add(aid)
                topL_unique.append(c)
                if len(topL_unique) >= self.topL:
                    break
        topL = topL_unique[: max(1, self.topL)] if topL_unique else cands[:1]

        if self.use_dynamic and self.selector is not None:
            agent, x, _st, raw_ms = self._select_agent_dynamic(topL, used_ids)
            match_score = float(raw_ms)
        else:
            # ✅ cold_start: static Top-L but round-robin across agents (no repeat)
            # Use step_index % len(topL) to cycle through topL candidates
            candidate_idx = step_index % len(topL) if topL else 0
            candidate = topL[candidate_idx]
            agent = candidate["agent"]
            match_score = float(candidate.get("match_score", 0.0))
            stt = self._agent_state(agent)
            x = self._build_x_from_candidate_or_fallback(
                candidate=candidate,
                agent=agent,
                dynamic_state=stt,
            )

        # ✅ P0-1: Use unified agent key resolution
        aid = self._resolve_agent_key(agent)
        if not aid:
            aid = f"agent_{id(agent)}"
        used_ids.add(aid)

        t0 = time.time()
        try:
            text = self._execute_subtask_on_agent(agent, st)
        except Exception as e:
            text = f"[AGENT_ERROR] {str(e)}"
        dt_ms = (time.time() - t0) * 1000.0

        rec = {
            "agent_id": aid,
            "match_score": float(match_score),
            "x": x,
            "latency_ms": float(dt_ms),
            "text": text,
            "final": self._extract_final_from_text(text) or "",
        }

        # ✅ planner-step base update (ok - latency penalty)
        # Skip updates in eval_only (e.g. test phase) to avoid label/data leakage.
        if self.use_dynamic and self.selector is not None and not self.eval_only and isinstance(x, list):
            ok = (text.strip() != "") and (not text.startswith("[ERROR]")) and (not text.startswith("[AGENT_ERROR]"))
            lat_norm = min(1.0, dt_ms / max(1.0, self.latency_scale_ms))
            penalty = self.latency_penalty * (lat_norm ** 0.5)
            base_reward = (1.0 if ok else 0.0) - penalty
            self.selector.update(x, base_reward)

        return rec

    def _run_plan_chain_v1(self, base_task: str, chain: List[Dict[str, Any]], base_ctx: Dict[str, Any]) -> Tuple[
        str, float, Dict[str, Any]]:
        used_ids = set()
        prev: List[str] = []
        w_sum = 0.0
        steps_trace: List[Dict[str, Any]] = []
        step_records: List[Dict[str, Any]] = []

        for step_idx, st in enumerate(chain):
            st2 = dict(st)
            st2["input"] = self._format_executor_input(base_task=base_task, q=str(st.get("input", "")), prev=prev)

            # inherit base context (including gold if provided)
            st2_ctx = dict(base_ctx or {})
            st2_ctx.update(st2.get("context", {}) or {})
            st2["context"] = st2_ctx

            rec = self._execute_one_subtask_beacon(st2, used_ids, base_ctx=st2_ctx, step_index=step_idx)
            txt = str(rec.get("text", ""))
            ms = float(rec.get("match_score", 0.0))

            prev.append(txt)
            w_sum += ms
            step_records.append(rec)
            steps_trace.append(
                {"subtask_id": st["id"], "meta": {"selected": rec.get("agent_id"), "match_score": ms}, "text": txt})

        w = w_sum / max(1, len(chain))
        final = prev[-1] if prev else ""
        return final, w, {"steps": steps_trace, "records": step_records}


# ---------------------- global API ----------------------
_global_orchestrator = SymphonyOrchestrator(
    verbose=False,
    use_dynamic=True,
    topL=3,
    linucb_alpha=1.0,
    linucb_l2=1.0,
    latency_scale_ms=2000.0,
    latency_penalty=0.2,
    win_bonus=0.5,
    correctness_bonus=0.0,  # 默认不启用 correctness reward（兼容旧 runner）
    incorrect_penalty=0.0,
    plan_k=1,  # 默认不启用 planner（BBH 先稳定评测）
    use_planner_decompose=True,  # ✅ All tasks go through prompt-based decomposition
    enable_risk_guard=False,
)


def init(
        *,
        verbose: Optional[bool] = None,
        use_dynamic: Optional[bool] = None,
        topL: Optional[int] = None,
        linucb_alpha: Optional[float] = None,
        linucb_l2: Optional[float] = None,
        plan_k: Optional[int] = None,
        use_planner_decompose: Optional[bool] = None,
        enable_risk_guard: Optional[bool] = None,
        # correctness reward knobs
        correctness_bonus: Optional[float] = None,
        incorrect_penalty: Optional[float] = None,
        dispatch_mode: Optional[str] = None,
        requester_id: Optional[str] = None,
        shared_timeout_s: Optional[float] = None,
        shared_poll_interval: Optional[float] = None,
        # ✅ P0-1: Cold-start priors injection
        priors: Optional[Dict[str, Dict[str, float]]] = None,
        priors_path: Optional[str] = None,
        # ✅ P0-D: Strict routing mode (experiment mode)
        strict_routing: Optional[bool] = None,
        # ✅ Eval-only: skip selector updates (e.g. test phase) to avoid leakage
        eval_only: Optional[bool] = None,
) -> None:
    """
    Configure global orchestrator WITHOUT clearing registered agents.
    """
    global _global_orchestrator

    if verbose is not None:
        _global_orchestrator.verbose = bool(verbose)

    if use_dynamic is not None:
        _global_orchestrator.use_dynamic = bool(use_dynamic)
        if _global_orchestrator.use_dynamic and _global_orchestrator.selector is None:
            a = float(linucb_alpha) if linucb_alpha is not None else 1.0
            l2 = float(linucb_l2) if linucb_l2 is not None else 1.0
            _global_orchestrator.selector = GlobalLinUCB(d=6, l2=l2, alpha=a)
        if (not _global_orchestrator.use_dynamic) and _global_orchestrator.selector is not None:
            _global_orchestrator.selector = None

    if topL is not None:
        _global_orchestrator.topL = max(1, int(topL))

    if linucb_alpha is not None or linucb_l2 is not None:
        if _global_orchestrator.use_dynamic:
            a = float(linucb_alpha) if linucb_alpha is not None else 1.0
            l2 = float(linucb_l2) if linucb_l2 is not None else 1.0
            _global_orchestrator.selector = GlobalLinUCB(d=6, l2=l2, alpha=a)

    if plan_k is not None:
        _global_orchestrator.plan_k = max(1, int(plan_k))
    if use_planner_decompose is not None:
        _global_orchestrator.use_planner_decompose = bool(use_planner_decompose)

    if enable_risk_guard is not None:
        _global_orchestrator.enable_risk_guard = bool(enable_risk_guard)
        if _global_orchestrator.enable_risk_guard and RiskAwareGuard is not None and RiskGuardConfig is not None:
            _global_orchestrator.risk_guard = RiskAwareGuard(RiskGuardConfig())
        else:
            _global_orchestrator.risk_guard = None

    if correctness_bonus is not None:
        _global_orchestrator.correctness_bonus = float(correctness_bonus)
    if incorrect_penalty is not None:
        _global_orchestrator.incorrect_penalty = float(incorrect_penalty)
    if dispatch_mode is not None:
        _global_orchestrator.dispatch_mode = str(dispatch_mode)

    if requester_id is not None:
        _global_orchestrator.requester_id = requester_id

    if shared_timeout_s is not None:
        _global_orchestrator.shared_timeout_s = float(shared_timeout_s)

    if shared_poll_interval is not None:
        _global_orchestrator.shared_poll_interval = float(shared_poll_interval)

    if strict_routing is not None:
        _global_orchestrator.strict_routing = bool(strict_routing)
    if eval_only is not None:
        _global_orchestrator.eval_only = bool(eval_only)

    if priors is not None or priors_path is not None:
        _priors: Dict[str, Dict[str, float]] = {}
        if priors is not None:
            _priors = priors
        elif priors_path:
            try:
                from core.cold_start import load_priors
                _priors = load_priors(priors_path)
                if _global_orchestrator.verbose:
                    print(f"[ORCHESTRATOR] Loaded priors from {priors_path} ({len(_priors)} agents)")
            except Exception as e:
                if _global_orchestrator.verbose:
                    print(f"[WARN] Failed to load priors from {priors_path}: {e}")
        
        # ✅ P0-1: Update orchestrator's priors and inject into existing agents (use unified key resolution)
        _global_orchestrator._learned_priors = _priors
        for agent in _global_orchestrator.agents:
            agent_key = _global_orchestrator._resolve_agent_key(agent)
            if agent_key and agent_key in _priors:
                agent.learned_priors = _priors[agent_key]
                bucket_count = len(_priors[agent_key])
                if _global_orchestrator.verbose:
                    print(f"[PRIORS] ✅ Injected agent={agent_key} buckets={bucket_count}")


def execute_task(
        task: Task,
        cot_count: int = 3,
        verbose: Optional[bool] = None,
        return_mode: str = "aggregate",
) -> Any:
    if verbose is not None:
        _global_orchestrator.verbose = bool(verbose)
    return _global_orchestrator.execute_task(task, cot_count=cot_count, return_mode=return_mode)


def register_agent(agent: Agent) -> None:
    _global_orchestrator.register_agent(agent)


def get_registered_agents() -> List[Agent]:
    return _global_orchestrator.get_registered_agents()


def set_eval_only(enable: bool) -> None:
    """Set eval-only mode: skip selector updates (e.g. test phase) to avoid leakage."""
    _global_orchestrator.eval_only = bool(enable)

