# protocol/task_contract.py
"""
Task contract and result definitions for distributed task execution.

✅ Symphony 2.0 changes (IMPORTANT):
1) Task must preserve `task_id` from incoming dict instead of generating a new one.
   - Otherwise task_id changes across hops/from_dict calls -> breaks bandit mapping.
2) Task must preserve `context` if present (so we can carry dispatch_id/task_key).
3) TaskResult includes `task_id` (dispatch_id / task_key) so requester can update LinUCB reliably.
4) Robust api_mode handling: from_dict prioritizes api_mode to avoid legacy/high-level mis-detection.
"""

import uuid
from typing import Dict, Any, List, Optional

__all__ = ["Task", "TaskResult"]


class TaskResult:
    """
    Task execution result container.

    ✅ Symphony 2.0:
    - add task_id: used as "dispatch_id / task_key" for requester-side bandit update
    """

    def __init__(
        self,
        target_id: str,
        executer_id: str,
        result: Any,
        previous_results: Any,
        task_id: str = "",
    ) -> None:
        self.target_id = target_id
        self.executer_id = executer_id  # keep legacy spelling
        self.result = result
        self.previous_results = previous_results
        self.task_id = task_id or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "executer_id": self.executer_id,  # keep legacy spelling
            "result": self.result,
            "previous_results": self.previous_results,
            "task_id": self.task_id,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TaskResult":
        # compat: accept both keys; still output executer_id
        exec_id = data.get("executer_id", "")
        if not exec_id:
            exec_id = data.get("executor_id", "")
        return TaskResult(
            target_id=data.get("target_id", ""),
            executer_id=exec_id,
            result=data.get("result", None),
            previous_results=data.get("previous_results", None),
            task_id=data.get("task_id", ""),
        )

    def __repr__(self) -> str:
        return f"<Result from {self.executer_id} to {self.target_id} task_id={self.task_id}>"


class Task:
    """
    Task contract for distributed computation.

    Supports:
    - High-level API (description/requirements/context/task_id)
    - Legacy API (subtask_id/steps/previous_results/original_problem/final_result/user_id)

    ✅ Symphony 2.0 fixes:
    - keep task_id stable (use provided task_id if present)
    - keep context (for dispatch_id/task_key carrying)
    - from_dict uses api_mode when available (avoid accidental mode flip)
    """

    def __init__(
        self,
        # New API
        description: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        # Legacy API
        subtask_id: Optional[int] = None,
        steps: Optional[Dict[str, Any]] = None,
        previous_results: Optional[List[Any]] = None,
        original_problem: Optional[str] = None,
        final_result: Optional[str] = None,
        user_id: Optional[str] = None,
        # Internal / compat
        api_mode: Optional[str] = None,  # "high_level" | "legacy" | None(auto)
    ) -> None:

        # Decide API mode
        forced = (api_mode or "").strip().lower()
        if forced not in ("high_level", "legacy", ""):
            forced = ""

        auto_high_level = (description is not None) or (requirements is not None)
        mode = forced or ("high_level" if auto_high_level else "legacy")

        # Always keep stable identifiers
        self.task_id = task_id or str(uuid.uuid4())
        self.context = context or ({"mode": "legacy"} if mode == "legacy" else {})

        if mode == "high_level":
            # ---------------- High-level ----------------
            self.description = description or ""
            self.requirements = requirements or []

            # legacy-compatible mirrors (keep if provided)
            self.subtask_id = subtask_id or 0
            self.steps = steps or {}
            self.previous_results = previous_results or []
            self.original_problem = original_problem or self.description
            self.final_result = final_result or ""
            self.user_id = user_id or "symphony_user"
            self._api_mode = "high_level"

        else:
            # ---------------- Legacy ----------------
            self.subtask_id = subtask_id or 0
            self.steps = steps or {}
            self.previous_results = previous_results or []
            self.original_problem = original_problem or ""
            self.final_result = final_result or ""
            self.user_id = user_id or "symphony_user"

            # best-effort new fields from legacy data
            self.description = self.original_problem or ""

            # ✅ FIX: extract real requirements from steps values
            reqs: List[str] = []
            if isinstance(self.steps, dict):
                for _, v in self.steps.items():
                    # typical: v = [instruction, requirement]
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        r = v[1]
                        if isinstance(r, str) and r.strip():
                            reqs.append(r.strip())
            self.requirements = reqs

            self._api_mode = "legacy"

    def to_dict(self) -> Dict[str, Any]:
        # Keep both sets of fields for compatibility, but api_mode must be stable.
        return {
            # legacy fields
            "subtask_id": self.subtask_id,
            "steps": self.steps,
            "previous_results": self.previous_results,
            "original_problem": self.original_problem,
            "final_result": self.final_result,
            "user_id": self.user_id,
            # stable routing fields
            "task_id": self.task_id,
            "context": self.context,
            "api_mode": getattr(self, "_api_mode", "legacy"),
            # high-level mirrors (safe to include; mode decided by api_mode in from_dict)
            "description": getattr(self, "description", ""),
            "requirements": getattr(self, "requirements", []),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Task":
        mode = str(data.get("api_mode", "") or "").strip().lower()

        # Priority 1: explicit api_mode
        if mode == "high_level":
            return Task(
                description=data.get("description"),
                requirements=data.get("requirements", []),
                context=data.get("context", {}),
                task_id=data.get("task_id"),
                subtask_id=data.get("subtask_id"),
                steps=data.get("steps"),
                previous_results=data.get("previous_results"),
                original_problem=data.get("original_problem"),
                final_result=data.get("final_result"),
                user_id=data.get("user_id"),
                api_mode="high_level",
            )

        if mode == "legacy":
            return Task(
                subtask_id=data.get("subtask_id", 0),
                steps=data.get("steps", {}),
                previous_results=data.get("previous_results", []),
                original_problem=data.get("original_problem", ""),
                final_result=data.get("final_result", ""),
                user_id=data.get("user_id", "symphony_user"),
                task_id=data.get("task_id"),
                context=data.get("context", {"mode": "legacy"}),
                api_mode="legacy",
            )

        # Priority 2: heuristic fallback (older payloads without api_mode)
        # If it has steps/subtask_id => legacy; else treat as high-level.
        if "steps" in data or "subtask_id" in data or "original_problem" in data:
            return Task(
                subtask_id=data.get("subtask_id", 0),
                steps=data.get("steps", {}),
                previous_results=data.get("previous_results", []),
                original_problem=data.get("original_problem", ""),
                final_result=data.get("final_result", ""),
                user_id=data.get("user_id", "symphony_user"),
                task_id=data.get("task_id"),
                context=data.get("context", {"mode": "legacy"}),
                api_mode="legacy",
            )

        return Task(
            description=data.get("description", ""),
            requirements=data.get("requirements", []),
            context=data.get("context", {}),
            task_id=data.get("task_id"),
            api_mode="high_level",
        )

    def __repr__(self) -> str:
        return f"<Task id={self.task_id} subtask_id={self.subtask_id} mode={getattr(self,'_api_mode','legacy')}>"

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)




