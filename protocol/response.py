# protocol/response.py
"""Beacon response handling for capability matching in symphony network."""

import uuid
import time
from typing import Dict, Any, Optional


class BeaconResponse:
    """Response to a Beacon message indicating capability match."""

    def __init__(
        self,
        responder_id: str,
        task_id: str,
        match_score: float = 1.0,
        estimate_cost: float = 1.0,

        # ---- Symphony 2.0 additions (optional for backward compatibility) ----
        available: Optional[bool] = None,                 # can accept task now
        dynamic_state: Optional[Dict[str, Any]] = None,   # load/latency/reputation/...
        capability_id: Optional[str] = None,              # optional: for ledger lookup
        timestamp: Optional[int] = None,
    ) -> None:
        self.response_id = str(uuid.uuid4())
        self.responder_id = responder_id
        self.task_id = task_id

        # keep 1.0 fields
        self.match_score = round(max(0.0, min(1.0, float(match_score))), 3)
        self.estimate_cost = round(max(0.0, float(estimate_cost)), 3)

        # new fields
        self.available = True if available is None else bool(available)
        self.dynamic_state = dynamic_state or {}
        self.capability_id = capability_id

        self.timestamp = int(time.time()) if timestamp is None else int(timestamp)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "response_id": self.response_id,
            "responder_id": self.responder_id,
            "task_id": self.task_id,
            "match_score": self.match_score,
            "estimate_cost": self.estimate_cost,
            "timestamp": self.timestamp,
        }

        data["available"] = self.available
        if self.dynamic_state:
            data["dynamic_state"] = self.dynamic_state
        if self.capability_id is not None:
            data["capability_id"] = self.capability_id

        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BeaconResponse":
        ds = data.get("dynamic_state")
        if not isinstance(ds, dict):
            ds = {}

        resp = BeaconResponse(
            responder_id=data.get("responder_id", "unknown"),
            task_id=data.get("task_id", ""),
            match_score=data.get("match_score", 1.0),
            estimate_cost=data.get("estimate_cost", 1.0),
            available=data.get("available", True),
            dynamic_state=ds,
            capability_id=data.get("capability_id", None),
            timestamp=data.get("timestamp", int(time.time())),
        )
        resp.response_id = data.get("response_id", str(uuid.uuid4()))
        return resp

    def __repr__(self) -> str:
        ds = self.dynamic_state or {}
        load = ds.get("load", None)
        lat = ds.get("latency_ms", None)
        return (
            f"<Response {self.response_id[:6]} from {self.responder_id}, "
            f"score={self.match_score}, cost={self.estimate_cost}, "
            f"avail={self.available}, load={load}, lat_ms={lat}>"
        )

