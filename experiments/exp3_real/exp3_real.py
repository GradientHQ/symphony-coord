#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, sys, json, time, math, random, argparse
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.agent import Agent
from core.embeddings import sim_emb
from core.linucb_selector import GlobalLinUCB


# -------------------------
# utils
# -------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def now() -> float:
    return time.time()

def ema(old: float, new: float, alpha: float) -> float:
    return (1 - alpha) * old + alpha * new

def is_success(text: str) -> int:
    if not text:
        return 0
    s = text.strip()
    if not s:
        return 0
    if s.startswith("[ERROR]") or s.startswith("[AGENT_ERROR]"):
        return 0
    return 1

def shaped_reward(cfg: Dict[str, Any], ok: int, latency_ms: float, token_out: int) -> float:
    rw = cfg["reward"]["shaped"]
    latency_scale = float(rw.get("latency_scale_ms", 2000.0))
    token_scale = float(rw.get("token_scale", 2000.0))
    lam = float(rw.get("lambda", 0.2))
    mu = float(rw.get("mu", 0.1))

    lat_pen = min(3.0, latency_ms / max(1e-9, latency_scale))
    tok_pen = min(3.0, token_out / max(1e-9, token_scale))
    return float(ok) - lam * lat_pen - mu * tok_pen


@dataclass
class AgentStats:
    reputation: float = 0.5
    latency_ms: float = 800.0
    available: int = 1
    load: float = 0.0


# -------------------------
# task pool
# -------------------------
def load_task_pool(cfg: Dict[str, Any], seed: int, n: int) -> List[Dict[str, Any]]:
    tp = cfg["task_pool"]
    assert tp["mode"] == "real_replay"
    real = tp["real"]
    data_path = real["data_path"]
    bench = real.get("benchmark_filter")
    rep = bool(real.get("sample_with_replacement", True))

    all_tasks: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if bench and obj.get("benchmark") != bench:
                continue
            all_tasks.append(obj)

    if not all_tasks:
        raise RuntimeError(f"No tasks after filter benchmark={bench}, file={data_path}")

    rng = random.Random(seed)
    if rep:
        return [rng.choice(all_tasks) for _ in range(n)]
    rng.shuffle(all_tasks)
    return all_tasks[:n]


# -------------------------
# agent loading
# -------------------------
def load_agents(cfg: Dict[str, Any]) -> Dict[str, Agent]:
    al = cfg["agent_loading"]
    config_dir = al["config_dir"]
    agent_ids = al["agent_ids"]
    letters = al["letter_order"]
    pat = al.get("file_pattern", "agent_{id}.yaml")

    if len(agent_ids) != len(letters):
        raise ValueError("agent_ids length must equal letter_order length")

    out: Dict[str, Agent] = {}
    for aid, letter in zip(agent_ids, letters):
        path = os.path.join(config_dir, pat.format(id=aid))
        with open(path, "r", encoding="utf-8") as f:
            agent_cfg = yaml.safe_load(f)
        out[letter] = Agent(config=agent_cfg)
    return out


# -------------------------
# Stage-A: Top-L candidates by composite score
# -------------------------
def stageA_topL(cfg: Dict[str, Any],
                agents: Dict[str, Agent],
                stats: Dict[str, AgentStats],
                task: Dict[str, Any]) -> Tuple[List[str], List[float], Dict[str, float]]:
    routing = cfg["routing"]
    topL = int(routing["topL"])
    w = routing.get("stageA_weights", {}) or {}
    lat_div = float(routing["features"]["latency_norm_div"])

    sim_map: Dict[str, float] = {}
    for letter, ag in agents.items():
        # sim_emb 已经处理 fallback=0.5
        sim_map[letter] = float(sim_emb(ag, task))

    def score(letter: str) -> float:
        st = stats[letter]
        latency_norm = min(3.0, float(st.latency_ms) / max(1e-9, lat_div))
        return (
            float(w.get("match", 1.0)) * sim_map[letter]
            + float(w.get("reputation", 0.5)) * float(st.reputation)
            + float(w.get("available", 0.2)) * float(st.available)
            - float(w.get("latency", 0.2)) * latency_norm
            - float(w.get("load", 0.0)) * float(st.load)
        )

    letters = list(agents.keys())
    letters.sort(key=score, reverse=True)
    cands = letters[:max(1, topL)]
    cand_scores = [score(c) for c in cands]
    return cands, cand_scores, sim_map


# -------------------------
# x vector for LinUCB (d=6)
# schema: [bias, match_score, load, latency_norm, reputation, available]
# -------------------------
def build_x_vec(cfg: Dict[str, Any], *,
                match_score: float,
                load: float,
                latency_ms: float,
                reputation: float,
                available: int) -> List[float]:
    lat_div = float(cfg["routing"]["features"]["latency_norm_div"])
    latency_norm = min(3.0, float(latency_ms) / max(1e-9, lat_div))
    return [1.0, float(match_score), float(load), float(latency_norm), float(reputation), float(available)]


# -------------------------
# policy pick
# -------------------------
def pick_executor(policy: str,
                  rng: random.Random,
                  all_letters: List[str],
                  cand_letters: List[str],
                  cand_scores: List[float],
                  linucb: Optional[GlobalLinUCB],
                  x_map: Dict[str, List[float]]) -> str:
    if policy == "linucb":
        assert linucb is not None
        # GlobalLinUCB in your Agent uses select(candidates_x: List[Tuple[id,x]])
        candidates_x = [(cid, x_map[cid]) for cid in cand_letters if cid in x_map]
        return linucb.select(candidates_x)

    if policy == "static_rule":
        best_i = max(range(len(cand_letters)), key=lambda i: cand_scores[i])
        return cand_letters[best_i]

    if policy == "random_topL":
        return rng.choice(cand_letters)

    if policy == "random":
        return rng.choice(all_letters)

    raise ValueError(f"Unknown policy: {policy}")


# -------------------------
# call agent (OpenRouter/OpenAI compat)
# -------------------------
def call_agent(cfg: Dict[str, Any], agent: Agent, task: Dict[str, Any]) -> Tuple[str, float, int, Optional[str]]:
    """
    Returns: (text, latency_ms, completion_tokens, error_type)
    """
    decoding = cfg["decoding"]
    bench = task.get("benchmark", "unknown")
    max_out = int(decoding["max_output_tokens"].get(bench, 256))

    prompt = (
        task.get("input")
        or task.get("question")
        or task.get("description")
        or json.dumps(task, ensure_ascii=False)
    )

    t0 = now()
    error_type = None
    completion_tokens = 0
    text = ""

    # Use structured generate_with_metadata (your OpenAICompatModel has it)
    if agent.base_model is None:
        text = "[AGENT_ERROR] base_model is None"
        error_type = "logic_error"
    else:
        try:
            res = agent.base_model.generate_with_metadata(
                prompt,
                max_tokens=max_out,
                temperature=float(decoding.get("temperature", 0.2)),
                top_p=float(decoding.get("top_p", 0.95)),
            )
            if res.get("success", False):
                text = res.get("response", "") or ""
                completion_tokens = int(res.get("completion_tokens", 0) or 0)
            else:
                error_type = res.get("error_type", "unknown_error")
                # 这里按你实验定义：输出为空视作 fail
                text = ""
        except Exception as e:
            error_type = "exception"
            text = ""

    latency_ms = (now() - t0) * 1000.0
    return text, latency_ms, completion_tokens, error_type


# -------------------------
# run one policy
# -------------------------
def run_one_policy(cfg: Dict[str, Any],
                   policy: str,
                   tasks: List[Dict[str, Any]],
                   agents: Dict[str, Agent],
                   outdir: str,
                   seed: int) -> None:
    rng = random.Random(seed + (abs(hash(policy)) % 100000))

    # stats init
    alpha = float(cfg["runtime"]["stats_ema_alpha"])
    lat_init = float(cfg["runtime"].get("latency_ema_init_ms", 800.0))
    stats: Dict[str, AgentStats] = {k: AgentStats(latency_ms=lat_init) for k in agents.keys()}

    # linucb init (only for linucb policy)
    linucb = None
    if policy == "linucb":
        lc = cfg["linucb"]
        linucb = GlobalLinUCB(d=6, l2=float(lc.get("l2", 1.0)), alpha=float(lc.get("alpha", 1.0)))

    fb = cfg.get("fallback", {}) or {}
    fb_on = bool(fb.get("enabled", True))
    max_retries = int(fb.get("max_retries", 2))
    update_mode = str(fb.get("bandit_update_mode", "final"))

    all_letters = list(agents.keys())
    logs: List[Dict[str, Any]] = []

    for t, task in enumerate(tasks, start=1):
        cand_letters, cand_scores, sim_map = stageA_topL(cfg, agents, stats, task)

        # prepare x_map for candidates
        x_map: Dict[str, List[float]] = {}
        for c in cand_letters:
            st = stats[c]
            x_map[c] = build_x_vec(
                cfg,
                match_score=float(sim_map[c]),
                load=float(st.load),
                latency_ms=float(st.latency_ms),
                reputation=float(st.reputation),
                available=int(st.available),
            )

        attempt = 0
        tried: List[str] = []
        final_ok = 0
        final_lat = 0.0
        final_tok = 0
        final_r = 0.0
        final_err = None
        final_chosen = None

        cur_cands = list(cand_letters)
        cur_scores = list(cand_scores)

        while True:
            attempt += 1
            chosen = pick_executor(policy, rng, all_letters, cur_cands, cur_scores, linucb, x_map)
            final_chosen = chosen
            tried.append(chosen)

            text, lat_ms, tok_out, err_type = call_agent(cfg, agents[chosen], task)
            ok = is_success(text)
            r = shaped_reward(cfg, ok, lat_ms, tok_out)

            # update runtime EMA stats (for Stage-A priors)
            st = stats[chosen]
            st.reputation = ema(st.reputation, float(ok), alpha)
            st.latency_ms = ema(st.latency_ms, float(lat_ms), alpha)

            # availability：你可以更细，但这里按“失败/超时/5xx”收敛得更快
            if ok == 1:
                st.available = 1
            else:
                if err_type in ("timeout", "server_500", "payment_required"):
                    st.available = 0

            # linucb update (only linucb policy)
            if policy == "linucb" and linucb is not None and chosen in x_map:
                do_update = (update_mode == "each") or (update_mode == "final" and (ok == 1 or attempt > max_retries))
                if do_update:
                    linucb.update(x_map[chosen], float(r))

            # finalize attempt
            final_ok, final_lat, final_tok, final_r, final_err = ok, lat_ms, tok_out, r, err_type

            # stop?
            if ok == 1:
                break
            if (not fb_on) or attempt > max_retries:
                break

            # fallback: remove chosen from current candidates if possible
            if chosen in cur_cands and len(cur_cands) > 1:
                idx = cur_cands.index(chosen)
                cur_cands.pop(idx)
                cur_scores.pop(idx)
                x_map.pop(chosen, None)
            else:
                break

        logs.append({
            "t": t,
            "policy": policy,
            "task_id": task.get("task_id", task.get("id", t)),
            "benchmark": task.get("benchmark"),
            "cand_letters": cand_letters,
            "cand_scores": cand_scores,
            "chosen": final_chosen,
            "attempt": attempt,
            "tried": tried,
            "ok": final_ok,
            "latency_ms": final_lat,
            "token_out": final_tok,
            "reward": final_r,
            "error_type": final_err,
        })

    dump_jsonl(os.path.join(outdir, f"{policy}.steps.jsonl"), logs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(_THIS_DIR, "config_exp3.yaml"))
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    if args.seed is not None:
        cfg["exp"]["seed"] = int(args.seed)
    if args.n is not None:
        cfg["exp"]["n_tasks"] = int(args.n)

    seed = int(cfg["exp"]["seed"])
    n_tasks = int(cfg["exp"]["n_tasks"])

    outdir = args.outdir or cfg["output"]["outdir"]
    ensure_dir(outdir)

    agents = load_agents(cfg)
    tasks = load_task_pool(cfg, seed=seed, n=n_tasks)

    if cfg["output"].get("save_sampled_tasks", True):
        dump_jsonl(os.path.join(outdir, "sampled_tasks.jsonl"), tasks)

    policies = [p["name"] for p in cfg["policies"]]
    for p in policies:
        run_one_policy(cfg, p, tasks, agents, outdir, seed)

    print(f"[OK] Exp3-real done. outdir={outdir}")


if __name__ == "__main__":
    main()
