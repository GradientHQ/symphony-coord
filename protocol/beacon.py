# protocol/beacon.py
import uuid
import time
from typing import Dict, Optional, Any, List


class Beacon:
    """Beacon message for discovering and requesting services in the network."""

    def __init__(
        self,
        sender: str,
        requirement: str,
        task_id: Optional[str] = None,
        ttl: int = 2,

        # ---- Symphony 2.0 additions (all optional for backward compatibility) ----
        task_type: Optional[str] = None,                      # e.g., "math", "code", "medical"
        requirement_embedding: Optional[List[float]] = None,  # task embedding (fixed dim)
        deadline_ms: Optional[int] = None,                    # QoS constraint
        priority: Optional[int] = None,                       # e.g., 0~10
        risk_level: Optional[str] = None,                     # "low"|"medium"|"high"
        max_candidates: Optional[int] = None,                 # stop after K replies (collector side)
        metadata: Optional[Dict[str, Any]] = None,            # extensible extra info
        timestamp: Optional[int] = None,                      # optional override
        beacon_id: Optional[str] = None,                      # optional override
    ) -> None:
        self.beacon_id = str(beacon_id or uuid.uuid4())
        self.sender = sender
        self.requirement = requirement

        # ✅ important: in Symphony2.0 you usually pass dispatch_id/task_key here
        self.task_id = str(task_id) if task_id else f"beacon:{self.beacon_id}"

        # ✅ normalize types
        try:
            self.ttl = max(0, int(ttl))
        except Exception:
            self.ttl = 2

        self.timestamp = int(time.time()) if timestamp is None else int(timestamp)

        # new fields
        self.task_type = task_type
        self.requirement_embedding = requirement_embedding
        self.deadline_ms = int(deadline_ms) if deadline_ms is not None else None
        self.priority = int(priority) if priority is not None else None
        self.risk_level = risk_level
        self.max_candidates = int(max_candidates) if max_candidates is not None else None
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "beacon_id": self.beacon_id,
            "sender": self.sender,
            "task_id": self.task_id,
            "requirement": self.requirement,
            "ttl": self.ttl,
            "timestamp": self.timestamp,
        }

        # add optional fields only when present (keeps payload small + backward compatible)
        if self.task_type is not None:
            data["task_type"] = self.task_type
        if self.requirement_embedding is not None:
            data["requirement_embedding"] = self.requirement_embedding
        if self.deadline_ms is not None:
            data["deadline_ms"] = self.deadline_ms
        if self.priority is not None:
            data["priority"] = self.priority
        if self.risk_level is not None:
            data["risk_level"] = self.risk_level
        if self.max_candidates is not None:
            data["max_candidates"] = self.max_candidates
        if self.metadata:
            data["metadata"] = self.metadata

        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Beacon":
        return Beacon(
            sender=data.get("sender", "unknown"),
            requirement=data.get("requirement", ""),
            task_id=data.get("task_id"),
            ttl=data.get("ttl", 2),

            # new optional fields
            task_type=data.get("task_type"),
            requirement_embedding=data.get("requirement_embedding"),
            deadline_ms=data.get("deadline_ms"),
            priority=data.get("priority"),
            risk_level=data.get("risk_level"),
            max_candidates=data.get("max_candidates"),
            metadata=data.get("metadata"),

            # overrides
            timestamp=data.get("timestamp"),
            beacon_id=data.get("beacon_id"),
        )

    def __repr__(self) -> str:
        tt = f", type={self.task_type}" if self.task_type else ""
        dl = f", deadline_ms={self.deadline_ms}" if self.deadline_ms else ""
        return (
            f"<Beacon {self.task_id[:16]} from {self.sender} "
            f"need '{self.requirement}' TTL={self.ttl}{tt}{dl}>"
        )

