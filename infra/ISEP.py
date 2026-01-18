# infra/ISEP.py
"""
Inter-node Service Exchange Protocol (ISEP) Client

✅ Shared-Blackboard + by-ref upgrade:
- shared: write beacon_response / task_result into shared blackboard
- shared: requester collects beacon responses from blackboard (instead of local pending_tasks)
- by-ref: delegate_task can store full task into blackboard, send only pointer over network
- executor: _handle_task resolves pointer -> loads real task from blackboard
- backward compatibility: if use_shared_bb=False, old behavior remains
"""

from typing import Dict, List, Any, Optional
import time
from queue import Queue

from protocol.beacon import Beacon
from protocol.response import BeaconResponse
from protocol.task_contract import TaskResult, Task
from infra.network_adapter import NetworkAdapter

from core.memory import get_blackboard, Blackboard


class ISEPClient:
    def __init__(
        self,
        node_id: str,
        network_adapter: NetworkAdapter,
        response_timeout: int = 1,
        use_shared_bb: bool = False,
        blackboard: Optional[Blackboard] = None,
    ):
        self.node_id = node_id
        self.network = network_adapter
        self.response_timeout = int(response_timeout)

        # ✅ shared blackboard switch
        self.use_shared_bb = bool(use_shared_bb)
        self.bb: Blackboard = blackboard or get_blackboard()

        # legacy: task_id -> list[response_dict]
        self.pending_tasks: Dict[str, List[Dict[str, Any]]] = {}

        self.beacon_queue = Queue()
        self.subtask_queue = Queue()
        self.task_result_queue = Queue()

        # handlers
        self.network.register_handler("beacon", self._handle_beacon)
        self.network.register_handler("beacon_response", self._handle_beacon_response)
        self.network.register_handler("task", self._handle_task)
        self.network.register_handler("task_result", self._handle_task_result)

    # ---------------------------------------------------------------------
    # broadcast beacon (trigger) then collect responses
    # shared: from blackboard
    # legacy: from local pending_tasks filled by _handle_beacon_response
    # ---------------------------------------------------------------------
    def broadcast_and_collect(self, beacon: Beacon) -> List[Dict[str, Any]]:
        tid = str(beacon.task_id)
        t0 = time.time()

        if not self.use_shared_bb:
            self.pending_tasks[tid] = []

        # still broadcast as trigger
        self.network.broadcast("beacon", beacon)
        time.sleep(self.response_timeout)

        if self.use_shared_bb:
            # Collect from blackboard events written after t0
            evs = self.bb.list_events(tid, count=500)
            replies: List[Dict[str, Any]] = []
            for e in evs:
                if e.kind != "beacon_response":
                    continue
                # filter old rounds (best-effort)
                try:
                    if float(e.ts) + 1e-6 < float(t0):
                        continue
                except Exception:
                    pass

                payload = e.payload if isinstance(e.payload, dict) else {"_raw": str(e.payload)}
                # extra safety: match task_id if present
                if str(payload.get("task_id", "")) and str(payload.get("task_id", "")) != tid:
                    continue
                replies.append(payload)
            return replies

        # legacy path
        replies = list(self.pending_tasks.get(tid, []))
        try:
            self.pending_tasks.pop(tid, None)
        except Exception:
            pass
        return replies

    def send_response(self, target_id: str, msg_type: str, response: Any) -> None:
        self.network.send(target_id, msg_type, response)

    # ---------------------------------------------------------------------
    # ✅ by-ref task delegation (full shared style)
    # shared + send_by_ref: store task into blackboard; send only pointer
    # legacy: send full task dict/object over network
    # ---------------------------------------------------------------------
    def delegate_task(self, executor_id: str, task: Task, send_by_ref: bool = False) -> str:
        if self.use_shared_bb and send_by_ref:
            # ✅ FIX: support dict task_id extraction
            if isinstance(task, dict):
                tid = str(task.get("task_id") or task.get("id") or "")
            else:
                tid = str(getattr(task, "task_id", "") or getattr(task, "id", "") or "")

            if not tid:
                # fallback: cannot by-ref without stable id
                self.network.send(executor_id, "task", task)
                return "ok"

            # best-effort serialize
            if isinstance(task, dict):
                task_dict = task
            else:
                task_dict = task.to_dict() if hasattr(task, "to_dict") else dict(getattr(task, "__dict__", {"_raw": str(task)}))

            bb_key = f"task:{tid}"
            # write into blackboard
            self.bb.kv_set(tid, bb_key, task_dict)
            self.bb.append_event(
                tid,
                "task_written",
                {"bb_key": bb_key, "executor": executor_id},
                cot_id="global",
                step_id="0",
            )

            # send only pointer
            self.network.send(executor_id, "task", {"task_id": tid, "bb_key": bb_key, "shared": True})
            return "ok"

        # legacy: send full task
        self.network.send(executor_id, "task", task)
        return "ok"

    # ---------------------------------------------------------------------
    # ✅ submit_result
    # shared: write TaskResult into blackboard (event + kv)
    # optionally also send over network (compat)
    # legacy: send over network only
    # ---------------------------------------------------------------------
    def submit_result(
        self,
        target_id: str,
        result: Any,
        previous_results: Any,
        task_id: str = "",
        also_send_network: bool = False,  # full shared 建议 False；兼容期可 True
    ) -> None:
        task_result = TaskResult(
            target_id=target_id,
            executer_id=self.node_id,
            result=result,
            previous_results=previous_results,
            task_id=task_id,
        )

        if self.use_shared_bb and task_id:
            res_dict = task_result.to_dict() if hasattr(task_result, "to_dict") else {"_raw": str(task_result)}
            tid = str(task_id)

            # event stream
            self.bb.append_event(
                tid,
                "task_result",
                res_dict,
                cot_id=str(res_dict.get("cot_id", "global")),
                step_id=str(res_dict.get("step_id", "0")),
            )
            # kv (requester polling uses result:<executor_id>)
            self.bb.kv_set(tid, f"result:{self.node_id}", res_dict)

            if not also_send_network:
                return

        # legacy / compatibility
        self.network.send(target_id, "task_result", task_result)

    # -------------------------- handlers --------------------------

    def _handle_beacon(self, sender_id: str, beacon: Any) -> None:
        self.beacon_queue.put((sender_id, "beacon", beacon))

    def _handle_beacon_response(self, sender_id: str, response: Any) -> None:
        # normalize to dict
        if isinstance(response, BeaconResponse):
            resp = response.to_dict()
        elif isinstance(response, dict):
            resp = response
        else:
            try:
                resp = response.to_dict()
            except Exception:
                return

        tid = str(resp.get("task_id", ""))

        # ✅ shared: write to blackboard events
        if self.use_shared_bb and tid:
            resp.setdefault("sender_id", sender_id)
            self.bb.append_event(
                tid,
                "beacon_response",
                resp,
                cot_id=str(resp.get("cot_id", "global")),
                step_id=str(resp.get("step_id", "0")),
            )

        # legacy: local buffer for broadcast_and_collect
        if tid and tid in self.pending_tasks:
            self.pending_tasks[tid].append(resp)

    def _handle_task(self, sender_id: str, task: Any) -> None:
        # by-ref pointer: {"task_id","bb_key","shared":True}
        if self.use_shared_bb and isinstance(task, dict) and task.get("shared") and task.get("task_id") and task.get("bb_key"):
            tid = str(task["task_id"])
            bb_key = str(task["bb_key"])
            real_task = self.bb.kv_get(tid, bb_key, default=task)
            self.subtask_queue.put((sender_id, "task", real_task))
            return

        self.subtask_queue.put((sender_id, "task", task))

    def _handle_task_result(self, sender_id: str, result: Any) -> None:
        # normalize to dict
        if isinstance(result, TaskResult):
            res = result.to_dict()
        elif isinstance(result, dict):
            res = result
        else:
            try:
                res = result.to_dict()
            except Exception:
                res = {"executer_id": sender_id, "result": str(result), "task_id": ""}

        tid = str(res.get("task_id", ""))

        # ✅ shared: write result into blackboard (event + kv)
        if self.use_shared_bb and tid:
            res.setdefault("sender_id", sender_id)
            self.bb.append_event(
                tid,
                "task_result",
                res,
                cot_id=str(res.get("cot_id", "global")),
                step_id=str(res.get("step_id", "0")),
            )
            # NOTE: if this came over network, sender_id is executor_id; keep consistent
            self.bb.kv_set(tid, f"result:{sender_id}", res)

        self.task_result_queue.put((sender_id, "task_result", res))

    # -------------------------- receive APIs --------------------------

    def receive_beacon(self, timeout=None):
        try:
            return self.beacon_queue.get(timeout=timeout)
        except Exception:
            return None, None, None

    def receive_task(self, timeout=None):
        try:
            return self.subtask_queue.get(timeout=timeout)
        except Exception:
            return None, None, None

    def receive_result(self, timeout=None):
        try:
            return self.task_result_queue.get(timeout=timeout)
        except Exception:
            return None, None, None
