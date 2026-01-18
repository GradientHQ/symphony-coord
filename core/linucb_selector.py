# core/linucb_selector.py
from __future__ import annotations
from typing import Dict, List, Tuple
import math


class GlobalLinUCB:
    """
    ✅ Symphony 2.0 Dynamic Beacon Selection (Global LinUCB)

    Maintain ONE global (A_t, b_t) as in the math proof:
      A_t = λ I + Σ x x^T
      b_t = Σ r x
      θ̂_t = A_t^{-1} b_t
      UCB(x) = x^T θ̂_t + β_t * sqrt(x^T A_t^{-1} x)

    Implementation:
    - keep A^{-1} via Sherman–Morrison update (no numpy)
    """

    def __init__(
        self,
        d: int,
        l2: float = 1.0,          # λ
        alpha: float = 1.0,       # exploration scale
        delta: float = 0.05,      # confidence
        S: float = 1.0,           # bound on ||θ*||
    ) -> None:
        self.d = int(d)
        self.l2 = float(l2)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.S = float(S)

        # A_inv = (1/λ) I  <=> A = λ I
        self.A_inv: List[List[float]] = [
            [(1.0 / self.l2 if i == j else 0.0) for j in range(self.d)]
            for i in range(self.d)
        ]
        self.b: List[float] = [0.0] * self.d
        self.t: int = 0

    def _dot(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _matvec(self, M: List[List[float]], v: List[float]) -> List[float]:
        out = [0.0] * self.d
        for i in range(self.d):
            s = 0.0
            row = M[i]
            for j in range(self.d):
                s += row[j] * v[j]
            out[i] = s
        return out

    def _outer(self, u: List[float], v: List[float]) -> List[List[float]]:
        return [[u[i] * v[j] for j in range(self.d)] for i in range(self.d)]

    def theta_hat(self) -> List[float]:
        return self._matvec(self.A_inv, self.b)

    def beta(self) -> float:
        """
        ✅ Standard LinUCB-style radius (one common form).
        You can adjust to match the exact theorem statement in your appendix.
        """
        t = max(1, self.t)
        # log det(A)/det(λI) upper bound: d*log(1 + t/λ)
        rad = math.sqrt(self.d * math.log(1.0 + t / self.l2) + 2.0 * math.log(1.0 / self.delta))
        return self.alpha * rad + math.sqrt(self.l2) * self.S

    def ucb(self, x: List[float]) -> float:
        th = self.theta_hat()
        mean = self._dot(th, x)
        Ax = self._matvec(self.A_inv, x)
        var = max(0.0, self._dot(x, Ax))
        return mean + self.beta() * math.sqrt(var)

    def select(self, candidates: List[Tuple[str, List[float]]]) -> str:
        best_id = candidates[0][0]
        best_score = -1e18
        for aid, x in candidates:
            s = self.ucb(x)
            if s > best_score:
                best_score = s
                best_id = aid
        return best_id

    def update(self, x: List[float], reward: float) -> None:
        """
        b <- b + r x
        A^{-1} <- A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        """
        r = float(reward)

        for i in range(self.d):
            self.b[i] += r * x[i]

        Ax = self._matvec(self.A_inv, x)
        denom = 1.0 + self._dot(x, Ax)
        if denom <= 1e-12:
            self.t += 1
            return

        outer = self._outer(Ax, Ax)
        for i in range(self.d):
            for j in range(self.d):
                self.A_inv[i][j] -= outer[i][j] / denom

        self.t += 1


def build_x(
    match_score: float,
    dynamic_state: Dict,
    available: bool,
    latency_scale_ms: float = 2000.0,
) -> List[float]:
    """
    ✅ Context vector x_{j,t} = g(r_t, c_j, z_{j,t})

    Minimal features (d=6):
      x = [1, match_score, load, latency_norm, reputation, available]

    ✅ Important:
    - normalize if ||x|| > 1 (bounded-context assumption used in proofs)
    """
    ms = max(0.0, min(1.0, float(match_score)))
    ds = dynamic_state if isinstance(dynamic_state, dict) else {}

    load = max(0.0, min(1.0, float(ds.get("load", 0.0))))
    lat_ms = max(0.0, float(ds.get("latency_ms", 500.0)))
    lat = max(0.0, min(1.0, lat_ms / max(1.0, latency_scale_ms)))
    rep = max(0.0, min(1.0, float(ds.get("reputation", 0.5))))
    av = 1.0 if bool(available) else 0.0

    x = [1.0, ms, load, lat, rep, av]

    norm = math.sqrt(sum(v * v for v in x))
    if norm > 1.0:
        x = [v / norm for v in x]
    return x


