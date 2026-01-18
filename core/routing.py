# core/routing.py
"""
✅ Symphony 2.0: Routing layer for candidate generation and feature construction.

Responsibilities:
- TopL candidate selection (composite score: sim_emb + prior_success + stability)
- Feature vector construction for LinUCB (build_x with ms=sim_emb, rep=prior_success)
- Bucket management (benchmark|difficulty_bin)
- Dynamic state extraction from agents

Design:
- TopL stage: composite score = w1*sim_emb + w2*prior_success + w3*stability
- UCB stage: ms=sim_emb, rep=prior_success, lat/load/av from dynamic_state

This avoids "always selecting the same agent" while leveraging warmup priors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple

# Lazy import embeddings (isolates implementation details)
try:
    from core.embeddings import sim_emb
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    sim_emb = None  # type: ignore

from core.linucb_selector import build_x


__all__ = [
    "make_bucket",
    "get_prior_success",
    "get_dynamic_state",
    "compute_stability_score",
    "compute_match_score_ms",
    "score_topL",
    "select_topL",
    "build_x_for_agent",
    "build_x_from_candidate",  # P1: Reuse TopL results
]


# -------------------------
# Bucket management
# -------------------------

def make_bucket(benchmark: str, difficulty_bin: str) -> str:
    """
    ✅ Create bucket key from benchmark and difficulty.
    
    Args:
        benchmark: e.g. "humaneval", "gsm8k"
        difficulty_bin: e.g. "hard", "simple", "easy"
    
    Returns:
        Bucket key: "benchmark|difficulty_bin"
    """
    bench = str(benchmark or "").strip()
    diff = str(difficulty_bin or "").strip() or "unknown"
    return f"{bench}|{diff}"


# -------------------------
# Prior success extraction
# -------------------------

def get_prior_success(
    agent: Any,
    benchmark: str = "",
    difficulty_bin: str = "",
    default: float = 0.5,
) -> float:
    """
    ✅ Get learned prior success rate for agent and bucket.
    
    Priority:
    1. agent.learned_priors[bucket] (from warmup)
    2. agent.match_hard / match_simple (config fallback)
    3. default (0.5)
    
    Args:
        agent: Agent object
        benchmark: e.g. "humaneval", "gsm8k"
        difficulty_bin: e.g. "hard", "simple", "easy"
        default: Default value if no prior found
    
    Returns:
        Prior success rate [0, 1]
    """
    # Try learned_priors first
    if benchmark and difficulty_bin:
        bucket = make_bucket(benchmark, difficulty_bin)
        lp = getattr(agent, "learned_priors", None)
        if lp is None and hasattr(agent, "__dict__"):
            lp = agent.__dict__.get("learned_priors")
        if isinstance(lp, dict) and bucket in lp:
            return max(0.0, min(1.0, float(lp[bucket])))
    
    # Fallback to config priors
    if difficulty_bin == "hard":
        ms = getattr(agent, "match_hard", None)
        if ms is None and hasattr(agent, "__dict__"):
            ms = agent.__dict__.get("match_hard")
        if ms is not None:
            return max(0.0, min(1.0, float(ms)))
    else:
        ms = getattr(agent, "match_simple", None)
        if ms is None and hasattr(agent, "__dict__"):
            ms = agent.__dict__.get("match_simple")
        if ms is not None:
            return max(0.0, min(1.0, float(ms)))
    
    return float(default)


# -------------------------
# Dynamic state extraction
# -------------------------

def get_dynamic_state(agent: Any) -> Dict[str, Any]:
    """
    ✅ Extract dynamic state from agent.
    
    Args:
        agent: Agent object (should have get_dynamic_state() method)
    
    Returns:
        Dict with load, latency_ms, reputation, available
    """
    if hasattr(agent, "get_dynamic_state"):
        try:
            st = agent.get_dynamic_state()
            if isinstance(st, dict):
                return {
                    "load": float(st.get("load", 0.0)),
                    "latency_ms": float(st.get("latency_ms", 500.0)),
                    "reputation": float(st.get("reputation", 0.5)),
                    "available": bool(st.get("available", True)),
                }
        except Exception:
            pass
    
    # Fallback defaults
    return {
        "load": 0.0,
        "latency_ms": 500.0,
        "reputation": 0.5,
        "available": True,
    }


# -------------------------
# Stability score computation
# -------------------------

def compute_stability_score(
    dynamic_state: Dict[str, Any],
    latency_scale_ms: float = 2000.0,
) -> float:
    """
    ✅ Compute stability score from dynamic state.
    
    stability = f(load, latency, availability)
    Lower load + lower latency + available = higher stability
    
    Args:
        dynamic_state: Dict with load, latency_ms, available
        latency_scale_ms: Latency normalization scale
    
    Returns:
        Stability score [0, 1]
    """
    load = max(0.0, min(1.0, float(dynamic_state.get("load", 0.0))))
    latency_ms = max(0.0, float(dynamic_state.get("latency_ms", 500.0)))
    available = bool(dynamic_state.get("available", True))
    
    lat_norm = max(0.0, min(1.0, latency_ms / max(1.0, latency_scale_ms)))
    
    # Lower is better for load and latency
    stability = (1.0 - load) * 0.4 + (1.0 - lat_norm) * 0.4 + (1.0 if available else 0.0) * 0.2
    return max(0.0, min(1.0, stability))


# -------------------------
# Match score for build_x (UCB stage)
# -------------------------

def compute_match_score_ms(sim_emb: float) -> float:
    """
    ✅ Compute match_score for build_x (UCB stage).
    
    Recommendation: ms = sim_emb (direct mapping, no composition)
    This avoids double-counting prior_success (which goes into reputation).
    
    Args:
        sim_emb: Embedding similarity [0, 1]
    
    Returns:
        Match score [0, 1]
    """
    return max(0.0, min(1.0, float(sim_emb)))


# -------------------------
# TopL composite score
# -------------------------

def score_topL(
    *,
    sim_emb: float,
    prior_success: float,
    stability: float,
    w1: float = 0.5,
    w2: float = 0.35,
    w3: float = 0.15,
) -> float:
    """
    ✅ Compute TopL composite score.
    
    score_topL = w1 * sim_emb + w2 * prior_success + w3 * stability
    
    Args:
        sim_emb: Embedding similarity [0, 1]
        prior_success: Learned prior success rate [0, 1] (from warmup)
        stability: Stability score [0, 1] (based on load, latency, availability)
        w1, w2, w3: Weights (should sum to ~1.0, but not strictly enforced)
    
    Returns:
        Composite score [0, 1]
    """
    score = (
        w1 * max(0.0, min(1.0, float(sim_emb)))
        + w2 * max(0.0, min(1.0, float(prior_success)))
        + w3 * max(0.0, min(1.0, float(stability)))
    )
    return max(0.0, min(1.0, score))


# -------------------------
# TopL candidate selection
# -------------------------

def select_topL(
    agents: List[Any],
    subtask: Dict[str, Any],
    topL: int,
    latency_scale_ms: float = 2000.0,
    use_embedding: bool = True,
) -> List[Dict[str, Any]]:
    """
    ✅ Select TopL candidates based on composite score.
    
    Args:
        agents: List of agent objects
        subtask: Subtask dict with "requirement", "benchmark", "difficulty_bin", etc.
        topL: Number of top candidates to return
        latency_scale_ms: Latency normalization scale
        use_embedding: If True, use embedding similarity; else fallback to capability_manager.match()
    
    Returns:
        List of candidate dicts:
        [
            {
                "agent": Agent,
                "match_score": float,  # TopL composite score
                "sim_emb": float,      # For UCB stage (ms = sim_emb)
                "prior_success": float # For UCB stage (rep = prior_success)
            },
            ...
        ]
        Sorted by match_score (descending)
    """
    req = str(subtask.get("requirement", "general-reasoning")).strip()
    bench = str(subtask.get("benchmark", "")).strip()
    diff = str(subtask.get("difficulty_bin", "")).strip() or "unknown"
    
    candidates: List[Dict[str, Any]] = []
    
    for agent in agents:
        # Compute sim_emb
        sim_emb_val = 0.5  # Default neutral similarity
        
        if use_embedding and HAS_EMBEDDINGS and sim_emb is not None:
            try:
                sim_emb_val = sim_emb(agent, subtask)
            except Exception:
                # ✅ C: If embedding fails, keep neutral (0.5), don't fallback to string match
                sim_emb_val = 0.5  # Maintain neutral, don't silently revert to old logic
        elif not use_embedding:
            # Only use capability_manager.match() if use_embedding=False explicitly
            if hasattr(agent, "capability_manager"):
                try:
                    sim_emb_val = float(agent.capability_manager.match(req))
                except Exception:
                    sim_emb_val = 0.5
            else:
                sim_emb_val = 0.5
        # else: use_embedding=True but HAS_EMBEDDINGS=False -> keep default 0.5
        
        # Get prior_success
        prior_success = get_prior_success(agent, bench, diff, default=0.5)
        
        # Get dynamic state and compute stability
        dynamic_state = get_dynamic_state(agent)
        stability = compute_stability_score(dynamic_state, latency_scale_ms=latency_scale_ms)
        
        # Compute TopL composite score
        topL_score = score_topL(
            sim_emb=sim_emb_val,
            prior_success=prior_success,
            stability=stability,
        )
        
        candidates.append({
            "agent": agent,
            "match_score": topL_score,  # TopL composite score
            "sim_emb": sim_emb_val,     # For UCB stage
            "prior_success": prior_success,  # For UCB stage
        })
    
    # Sort by match_score (descending) and return topL
    candidates.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)
    return candidates[:max(1, int(topL))]


# -------------------------
# Feature vector construction for LinUCB
# -------------------------

def build_x_for_agent(
    agent: Any,
    subtask: Dict[str, Any],
    latency_scale_ms: float = 2000.0,
    use_embedding: bool = True,
) -> Tuple[List[float], float, float]:
    """
    ✅ Build feature vector x for LinUCB (d=6).
    
    x = [1, ms, load, lat, rep, av]
    where:
    - ms = sim_emb (embedding similarity)
    - rep = prior_success (learned prior from warmup)
    - lat/load/av from dynamic_state
    
    This ensures prior_success is not double-counted (only in rep, not in ms).
    
    Args:
        agent: Agent object
        subtask: Subtask dict
        latency_scale_ms: Latency normalization scale
        use_embedding: If True, use embedding similarity; else fallback
    
    Returns:
        Tuple of (x, sim_emb, prior_success):
        - x: Feature vector [1, ms, load, lat, rep, av]
        - sim_emb: Embedding similarity (for logging)
        - prior_success: Prior success rate (for logging)
    """
    req = str(subtask.get("requirement", "general-reasoning")).strip()
    bench = str(subtask.get("benchmark", "")).strip()
    diff = str(subtask.get("difficulty_bin", "")).strip() or "unknown"
    
    # Compute sim_emb (for ms)
    sim_emb_val = 0.5  # Default neutral similarity
    
    if use_embedding and HAS_EMBEDDINGS and sim_emb is not None:
        try:
            sim_emb_val = sim_emb(agent, subtask)
        except Exception:
            # ✅ C: If embedding fails, keep neutral (0.5), don't fallback to string match
            sim_emb_val = 0.5
    elif not use_embedding:
        # Only use capability_manager.match() if use_embedding=False explicitly
        if hasattr(agent, "capability_manager"):
            try:
                sim_emb_val = float(agent.capability_manager.match(req))
            except Exception:
                sim_emb_val = 0.5
        else:
            sim_emb_val = 0.5
    # else: use_embedding=True but HAS_EMBEDDINGS=False -> keep default 0.5
    
    # Get prior_success (for reputation)
    prior_success = get_prior_success(agent, bench, diff, default=0.5)
    
    # Get dynamic state
    dynamic_state = get_dynamic_state(agent)
    
    # Build x: ms = sim_emb, rep = prior_success
    ms = compute_match_score_ms(sim_emb_val)
    x = build_x(
        match_score=ms,  # ✅ ms = sim_emb
        dynamic_state={
            "load": dynamic_state["load"],
            "latency_ms": dynamic_state["latency_ms"],
            "reputation": prior_success,  # ✅ rep = prior_success
        },
        available=dynamic_state["available"],
        latency_scale_ms=latency_scale_ms,
    )
    
    return x, sim_emb_val, prior_success


# -------------------------
# Feature vector from candidate (P1: reuse TopL results)
# -------------------------

def build_x_from_candidate(
    candidate: Dict[str, Any],
    agent: Optional[Any] = None,
    latency_scale_ms: float = 2000.0,
) -> Tuple[List[float], float, float]:
    """
    ✅ P1: Build feature vector x from candidate dict (reuse TopL results).
    
    This avoids recomputing embedding/similarity between select_topL and UCB stage,
    ensuring consistency and reducing cost.
    
    Interface: Only depends on candidate + dynamic_state (recomputed).
    
    Args:
        candidate: Candidate dict from select_topL() with fixed schema:
            - "agent": Agent object (required)
            - "sim_emb": float (from TopL stage, required)
            - "prior_success": float (from TopL stage, required)
            - "match_score": float (TopL composite score, optional, for logging)
        agent: Agent object (if None, uses candidate["agent"])
        latency_scale_ms: Latency normalization scale
    
    Returns:
        Tuple of (x, sim_emb_val, prior_success_val):
        - x: Feature vector [1, ms, load, lat, rep, av]
        - sim_emb_val: Embedding similarity (for logging)
        - prior_success_val: Prior success rate (for logging)
    """
    # ✅ Ensure candidate schema consistency
    ag = agent or candidate.get("agent")
    if ag is None:
        # Fallback: use defaults if agent missing
        x = build_x(
            match_score=0.5,
            dynamic_state={"load": 0.0, "latency_ms": 500.0, "reputation": 0.5},
            available=True,
            latency_scale_ms=latency_scale_ms,
        )
        return x, 0.5, 0.5
    
    # Reuse values from candidate (no recomputation of sim_emb or prior_success)
    sim_emb_val = float(candidate.get("sim_emb", 0.5))
    prior_success_val = float(candidate.get("prior_success", 0.5))
    
    # ✅ Get dynamic state (may have changed since TopL, so recompute)
    dynamic_state = get_dynamic_state(ag)
    
    # Build x: ms = sim_emb, rep = prior_success (from candidate)
    ms = compute_match_score_ms(sim_emb_val)
    x = build_x(
        match_score=ms,  # ✅ ms = sim_emb (from candidate)
        dynamic_state={
            "load": dynamic_state["load"],
            "latency_ms": dynamic_state["latency_ms"],
            "reputation": prior_success_val,  # ✅ rep = prior_success (from candidate)
        },
        available=dynamic_state["available"],
        latency_scale_ms=latency_scale_ms,
    )
    
    return x, sim_emb_val, prior_success_val
