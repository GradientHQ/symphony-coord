"""
Core memory module.

- LocalMemory: local cache + neighbor tracking (kept for backward compatibility).
- Blackboard: shared blackboard abstraction for cross-agent shared state.
  - InProcBlackboard: in-process implementation (single orchestrator process).
  - RedisBlackboard: real shared implementation across processes/machines via Redis.

Recommended:
- Keep LocalMemory for neighbor scoring history.
- Use Blackboard for task/run context, intermediate results, voting, traces, etc.
"""

from __future__ import annotations

import os
import json
import time
import uuid
from dataclasses import dataclass
from collections import deque, defaultdict
from threading import RLock
from typing import Dict, List, Any, Optional, Tuple

__all__ = [
    "LocalMemory",
    "BlackboardEvent",
    "Blackboard",
    "InProcBlackboard",
    "RedisBlackboard",
    "get_blackboard",
]


# ---------------------------------------------------------------------
# Existing LocalMemory (unchanged behavior)
# ---------------------------------------------------------------------

class LocalMemory:
    """Local memory manager for caching tasks and tracking neighbors.

    This class provides functionality to cache task results, maintain
    neighbor information, and track neighbor performance scores for
    efficient task delegation and collaboration.
    """

    def __init__(self, task_limit: int = 100, neighbor_limit: int = 50):
        self.task_cache = deque(maxlen=task_limit)
        self.neighbor_data = {}  # node_id -> {capabilities, last_seen, score}
        self.neighbor_scores = defaultdict(lambda: deque(maxlen=20))

    def store_result(self, result: Dict[str, Any]) -> None:
        self.task_cache.append({"timestamp": int(time.time()), **result})

    def cache_task(self, task_id: str, task_data: Dict) -> None:
        self.task_cache.append({"task_id": task_id, **task_data})

    def get_recent_tasks(self, n: int = 5) -> List[Dict]:
        return list(self.task_cache)[-n:]

    def update_neighbor(self, node_id: str, capabilities: List[str], success: bool) -> None:
        now = int(time.time())
        if node_id not in self.neighbor_data:
            self.neighbor_data[node_id] = {
                "capabilities": capabilities,
                "last_seen": now,
                "score": 0.5,
            }
        else:
            self.neighbor_data[node_id]["last_seen"] = now
            self.neighbor_data[node_id]["capabilities"] = capabilities

        self.neighbor_scores[node_id].append(1.0 if success else 0.0)

    def get_neighbors(self) -> List[str]:
        return list(self.neighbor_data.keys())

    def get_neighbor_score(self, node_id: str) -> float:
        scores = self.neighbor_scores[node_id]
        if not scores:
            return 0.5
        return round(sum(scores) / len(scores), 3)

    def dump(self) -> None:
        print("\n[LocalMemory Dump]")
        for task in self.task_cache:
            print("- Task:", task)
        for node_id, meta in self.neighbor_data.items():
            score = self.get_neighbor_score(node_id)
            print(f"- Neighbor {node_id[:6]} | Score: {score}")


# ---------------------------------------------------------------------
# Shared Blackboard (new)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class BlackboardEvent:
    ts: float
    run_id: str
    kind: str              # e.g. "user_task" | "plan" | "subtask" | "result" | "vote" | "note"
    cot_id: str            # "0" / "1" / "2" or "global"
    step_id: str           # "0","1","2"... (string for convenience)
    payload: Dict[str, Any]


class Blackboard:
    """
    Blackboard abstract interface.

    Two storage styles:
    - append_event(): append-only event log (for replay/debug)
    - kv_set/kv_get(): snapshot store (for fast reads)
    """

    def new_run(self) -> str:
        raise NotImplementedError

    def append_event(self, run_id: str, kind: str, payload: Dict[str, Any],
                     cot_id: str = "global", step_id: str = "0") -> str:
        raise NotImplementedError

    def list_events(self, run_id: str, count: int = 200) -> List[BlackboardEvent]:
        raise NotImplementedError

    def kv_set(self, run_id: str, key: str, value: Any) -> None:
        raise NotImplementedError

    def kv_get(self, run_id: str, key: str, default: Any = None) -> Any:
        raise NotImplementedError

    def acquire_step(self, run_id: str, cot_id: str, step_id: str, ttl_sec: int = 300) -> bool:
        """
        Optional concurrency guard to prevent duplicate execution across workers.
        """
        return True

    def render_context(self, run_id: str, cot_id: str, upto_step: int,
                       limit_events: int = 80) -> str:
        """
        Build compact textual context for prompting.
        """
        evs = self.list_events(run_id, count=limit_events * 3)

        lines: List[str] = []
        for e in evs:
            if e.cot_id not in (cot_id, "global"):
                continue
            # step_id is string; best effort parse
            try:
                sid = int(e.step_id)
            except Exception:
                continue
            if sid > upto_step:
                continue

            if e.kind == "user_task":
                lines.append(f"[USER_TASK] {e.payload.get('text','')}")
            elif e.kind == "plan":
                lines.append(f"[PLAN {e.cot_id}] {e.payload.get('text','')}")
            elif e.kind == "subtask":
                lines.append(f"[SUBTASK {e.cot_id}/{e.step_id}] {e.payload.get('q','')}")
            elif e.kind == "result":
                ans = e.payload.get("answer", "")
                score = e.payload.get("score", None)
                lines.append(f"[RESULT {e.cot_id}/{e.step_id}] ans={ans} score={score}")
            elif e.kind == "vote":
                lines.append(f"[VOTE] {e.payload}")
            elif e.kind == "note":
                lines.append(f"[NOTE] {e.payload.get('text','')}")
            else:
                lines.append(f"[{e.kind}] {e.payload}")

        # keep last N lines
        return "\n".join(lines[-limit_events:])


class InProcBlackboard(Blackboard):
    """
    In-process blackboard (shared within one Python process).
    Good for debugging and single-orchestrator runs.
    """

    def __init__(self, max_events_per_run: int = 5000):
        self._lock = RLock()
        self._events: Dict[str, List[BlackboardEvent]] = {}
        self._kv: Dict[str, Dict[str, Any]] = {}
        self._max = max_events_per_run

    def new_run(self) -> str:
        run_id = str(uuid.uuid4())
        with self._lock:
            self._events[run_id] = []
            self._kv[run_id] = {}
        return run_id

    def append_event(self, run_id: str, kind: str, payload: Dict[str, Any],
                     cot_id: str = "global", step_id: str = "0") -> str:
        ev = BlackboardEvent(
            ts=time.time(),
            run_id=run_id,
            kind=kind,
            cot_id=cot_id,
            step_id=step_id,
            payload=payload,
        )
        with self._lock:
            if run_id not in self._events:
                self._events[run_id] = []
                self._kv[run_id] = {}
            self._events[run_id].append(ev)
            if len(self._events[run_id]) > self._max:
                self._events[run_id] = self._events[run_id][-self._max:]

        # local "event id"
        return f"local:{run_id}:{len(self._events[run_id])}"

    def list_events(self, run_id: str, count: int = 200) -> List[BlackboardEvent]:
        with self._lock:
            evs = list(self._events.get(run_id, []))
        return evs[-count:]

    def kv_set(self, run_id: str, key: str, value: Any) -> None:
        with self._lock:
            if run_id not in self._kv:
                self._kv[run_id] = {}
            self._kv[run_id][key] = value

    def kv_get(self, run_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._kv.get(run_id, {}).get(key, default)


class RedisBlackboard(Blackboard):
    """
    Real shared blackboard via Redis.
    - Event log: Redis Stream (XADD/XRANGE)
    - KV store: Redis Hash (HSET/HGET)
    - Step lock: SET NX EX
    """

    def __init__(self, url: str, prefix: str = "bb", maxlen: int = 5000):
        try:
            import redis  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "RedisBlackboard requires 'redis' package. Install via: pip install redis"
            ) from e

        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._p = prefix
        self._maxlen = maxlen

    def _events_key(self, run_id: str) -> str:
        return f"{self._p}:{run_id}:events"

    def _kv_key(self, run_id: str) -> str:
        return f"{self._p}:{run_id}:kv"

    def _lock_key(self, run_id: str, cot_id: str, step_id: str) -> str:
        return f"{self._p}:{run_id}:lock:{cot_id}:{step_id}"

    def new_run(self) -> str:
        run_id = str(uuid.uuid4())
        # initialize kv hash to make run visible
        self._redis.hset(self._kv_key(run_id), mapping={"_created_ts": str(time.time())})
        return run_id

    def acquire_step(self, run_id: str, cot_id: str, step_id: str, ttl_sec: int = 300) -> bool:
        return bool(self._redis.set(self._lock_key(run_id, cot_id, step_id), "1", nx=True, ex=ttl_sec))

    def append_event(self, run_id: str, kind: str, payload: Dict[str, Any],
                     cot_id: str = "global", step_id: str = "0") -> str:
        fields = {
            "ts": str(time.time()),
            "run_id": run_id,
            "kind": kind,
            "cot_id": cot_id,
            "step_id": step_id,
            "payload": json.dumps(payload, ensure_ascii=False),
        }
        eid = self._redis.xadd(
            self._events_key(run_id),
            fields,
            maxlen=self._maxlen,
            approximate=True,
        )
        return str(eid)

    def list_events(self, run_id: str, count: int = 200) -> List[BlackboardEvent]:
        rows: List[Tuple[str, Dict[str, str]]] = self._redis.xrevrange(
            self._events_key(run_id), max="+", min="-", count=count
        )
        rows = list(reversed(rows))

        out: List[BlackboardEvent] = []
        for _id, f in rows:
            try:
                payload = json.loads(f.get("payload", "{}"))
            except Exception:
                payload = {"_raw_payload": f.get("payload", "")}

            out.append(
                BlackboardEvent(
                    ts=float(f.get("ts", "0") or "0"),
                    run_id=f.get("run_id", run_id),
                    kind=f.get("kind", "unknown"),
                    cot_id=f.get("cot_id", "global"),
                    step_id=f.get("step_id", "0"),
                    payload=payload,
                )
            )
        return out

    def kv_set(self, run_id: str, key: str, value: Any) -> None:
        self._redis.hset(self._kv_key(run_id), key, json.dumps(value, ensure_ascii=False))

    def kv_get(self, run_id: str, key: str, default: Any = None) -> Any:
        v = self._redis.hget(self._kv_key(run_id), key)
        if v is None:
            return default
        try:
            return json.loads(v)
        except Exception:
            return v


# ---------------------------------------------------------------------
# Blackboard factory (choose backend by env/config)
# ---------------------------------------------------------------------

_GLOBAL_BB: Optional[Blackboard] = None

def get_blackboard() -> Blackboard:
    """
    Factory:
    - If env BLACKBOARD_URL exists -> RedisBlackboard
    - Else -> InProcBlackboard
    """
    global _GLOBAL_BB
    if _GLOBAL_BB is not None:
        return _GLOBAL_BB

    url = os.environ.get("BLACKBOARD_URL", "").strip()
    if url:
        _GLOBAL_BB = RedisBlackboard(url=url, prefix=os.environ.get("BLACKBOARD_PREFIX", "bb"))
    else:
        _GLOBAL_BB = InProcBlackboard()

    return _GLOBAL_BB

