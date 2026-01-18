# agents/agent.py
"""
Agent implementation for distributed computation in Symphony Network.

✅ Symphony 2.0 Key Additions:
- dynamic_state reporting in BeaconResponse
- requester-side Dynamic Beacon Selection via LinUCB (GlobalLinUCB)
- strict task_key/dispatch_id propagation to guarantee bandit update hits

✅ Shared + by-ref FULL upgrade:
- pass use_shared_bb + blackboard into ISEPClient (requester collects beacon_response from BB)
- requester enforces task.task_id = dispatch_id (task_key) so by-ref & result routing are consistent
- delegate_task(..., send_by_ref=use_shared_bb) so task payload is stored in BB and only pointer is sent
- executor submit_result(..., also_send_network=not use_shared_bb) (shared defaults to BB-only)
- requester polls BB kv for results when use_shared_bb=True (no receive_result dependency)
- handle_beacon: shared path writes beacon_response into BB and returns; legacy path still works
"""

from typing import Dict, List, Tuple, Any, Optional
import time
import os
import json

from models.base_loader import BaseModel

try:
    import requests
except ImportError:
    requests = None

# -------------------- LinUCB selector --------------------
from core.linucb_selector import GlobalLinUCB, build_x

try:
    from protocol.beacon import Beacon
    from protocol.response import BeaconResponse
except ImportError:
    Beacon = None
    BeaconResponse = None

try:
    from core.capability import CapabilityManager
    from core.memory import LocalMemory, get_blackboard
except ImportError:
    # fallback for minimal test
    class CapabilityManager:
        def __init__(self, capabilities):
            self.capabilities = capabilities

        def match(self, requirement):
            return 0.5 if requirement in " ".join(self.capabilities) else 0.1

    class LocalMemory:
        def __init__(self):
            pass

    def get_blackboard():
        return None

try:
    from protocol.task_contract import TaskResult, Task
except ImportError:
    TaskResult = None
    Task = None

try:
    from infra.ISEP import ISEPClient
    from infra.network_adapter import NetworkAdapter
except ImportError:
    ISEPClient = None
    NetworkAdapter = None


def _is_openai_spec(spec: str) -> bool:
    return isinstance(spec, str) and spec.startswith("openai:")


def _parse_openai_spec(spec: str):
    rest = spec[len("openai:"):]
    if "#model=" in rest:
        api_base, model = rest.split("#model=", 1)
    else:
        api_base, model = rest, None
    api_base = api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base += "/v1"
    return api_base, model


def _is_openrouter_spec(spec: str) -> bool:
    return isinstance(spec, str) and spec.startswith("openrouter:")


def _parse_openrouter_spec(spec: str):
    """解析 openrouter:model 或 openrouter:api_base#model=xxx 格式"""
    rest = spec[len("openrouter:"):]
    if "#model=" in rest:
        api_base, model = rest.split("#model=", 1)
    else:
        # 默认使用 OpenRouter 官方 endpoint，rest 就是 model
        api_base = "https://openrouter.ai/api"
        model = rest or None
    api_base = api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base += "/v1"
    return api_base, model


class OpenAICompatModel:
    """Minimal OpenAI-compatible HTTP client (optional)."""

    def __init__(self, api_base: str, model: str, system_prompt: str = "", 
                 api_key: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None):
        # ========== 原代码（已注释） ==========
        # def __init__(self, api_base: str, model: str, system_prompt: str = ""):
        #     if requests is None:
        #         raise RuntimeError("requests not installed. `pip install requests`")
        #     self.api_base = api_base.rstrip("/")
        #     self.model = model
        #     self.system_prompt = system_prompt or ""
        #     self._url_chat = f"{self.api_base}/chat/completions"
        #     self._headers = {
        #         "Content-Type": "application/json",
        #         "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'EMPTY')}",
        #     }
        # ====================================
        if requests is None:
            raise RuntimeError("requests not installed. `pip install requests`")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt or ""
        self._url_chat = f"{self.api_base}/chat/completions"
        
        # 支持自定义 API key（默认仍用 OPENAI_API_KEY）
        api_key = api_key or os.getenv('OPENAI_API_KEY', 'EMPTY')
        
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        # 支持额外 headers（用于 OpenRouter 的 HTTP-Referer 等）
        if extra_headers:
            self._headers.update(extra_headers)

    def generate(self, text: str, max_tokens: int = 10124, temperature: float = 0.2, top_p: float = 0.9) -> str:
        """
        Generate text using API. Returns string on success.
        
        Note: For structured error handling, use generate_with_metadata() instead.
        This method maintains backward compatibility but may raise exceptions on errors.
        """
        result = self.generate_with_metadata(text, max_tokens, temperature, top_p)
        if result.get("success", False):
            return result.get("response", "")
        # For backward compatibility, raise exception on failure
        error_type = result.get("error_type", "unknown_error")
        error_msg = result.get("error_message", "API call failed")
        raise RuntimeError(f"{error_type}: {error_msg}")
    
    def generate_with_metadata(self, text: str, max_tokens: int = 10124, temperature: float = 0.2, top_p: float = 0.9) -> Dict[str, Any]:
        """
        Generate text using API with structured error handling.
        
        Returns:
            Dict with keys:
                - success: bool
                - response: str (if success=True)
                - error_type: str (if success=False): "payment_required", "server_500", "timeout", "logic_error", etc.
                - error_message: str (if success=False)
                - prompt_tokens: int
                - completion_tokens: int
                - total_tokens: int
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        try:
            r = requests.post(self._url_chat, headers=self._headers, data=json.dumps(payload), timeout=180)
            
            # Handle HTTP errors with structured error types
            if r.status_code == 402:
                # Payment Required - allow fallback
                error_data = {}
                try:
                    error_data = r.json()
                except:
                    pass
                return {
                    "success": False,
                    "error_type": "payment_required",
                    "error_message": error_data.get("error", {}).get("message", "Payment required"),
                    "response": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            elif r.status_code >= 500:
                # Server errors (500, 502, 503, etc.) - allow fallback
                error_data = {}
                try:
                    error_data = r.json()
                except:
                    pass
                return {
                    "success": False,
                    "error_type": "server_500",
                    "error_message": error_data.get("error", {}).get("message", f"Server error {r.status_code}"),
                    "response": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            elif not r.ok:
                # Other HTTP errors (400, 401, 403, 429, etc.)
                error_data = {}
                try:
                    error_data = r.json()
                except:
                    pass
                # 400/401/403 are typically logic errors (bad request, auth), don't allow fallback
                # 429 (rate limit) might allow fallback, but treat as logic_error for now
                error_type = "logic_error" if r.status_code < 500 else "server_500"
                return {
                    "success": False,
                    "error_type": error_type,
                    "error_message": error_data.get("error", {}).get("message", f"HTTP {r.status_code}"),
                    "response": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            
            # Success: parse response
            data = r.json()
            usage = data.get("usage", {})
            response_text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            
            return {
                "success": True,
                "response": response_text,
                "error_type": None,
                "error_message": None,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            
        except requests.exceptions.Timeout:
            # Timeout - allow fallback
            return {
                "success": False,
                "error_type": "timeout",
                "error_message": "Request timeout",
                "response": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        except requests.exceptions.RequestException as e:
            # Network errors - treat as logic_error (don't allow fallback)
            return {
                "success": False,
                "error_type": "logic_error",
                "error_message": str(e),
                "response": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        except Exception as e:
            # Other exceptions - treat as logic_error
            return {
                "success": False,
                "error_type": "logic_error",
                "error_message": str(e),
                "response": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def extract_task(self, user_input: str):
        return "", user_input.strip(), True

    def generate_task_dag(self, background: str, question: str, user_input: str, domain: str):
        steps = {"1": ("Solve the problem carefully", "general-reasoning")}
        return steps, True


class Agent:
    def __init__(
        self,
        node_id: str = None,
        capabilities: List[str] = None,
        system_prompt: str = None,
        base_model: str = None,
        gpu_id: int = 0,
        config: Dict[str, Any] = None,
    ) -> None:
        if config is not None:
            self._init_from_config(config)
        else:
            self._init_from_params(node_id, capabilities, system_prompt, base_model, gpu_id)

        # ✅ Symphony2.0: global LinUCB + pending buffer (requester side)
        self._selector = GlobalLinUCB(d=6, l2=1.0, alpha=float(getattr(self, "_linucb_alpha", 1.0)))
        self._pending: Dict[str, Dict[str, Any]] = {}  # dispatch_id -> {x,t0,executor}

    # ---------------- dynamic state ----------------
    def _init_dynamic_state(self, max_inflight: int = 1, latency_ema_init_ms: float = 500.0, latency_ema_beta: float = 0.2):
        self._inflight = 0
        self._max_inflight = max(1, int(max_inflight))
        self._latency_ema_ms = float(latency_ema_init_ms)
        self._latency_beta = float(latency_ema_beta)
        self._reputation = 0.5

    def _init_from_config(self, config: Dict[str, Any]) -> None:
        self.agent_id = config["node_id"]
        self.gpu_id = config.get("gpu_id", 0)
        self.system_prompt = config.get("sys_prompt", "You are a helpful AI assistant.")
        self.capabilities = config.get("capabilities", [])
        self._linucb_alpha = float(config.get("linucb_alpha", 1.0))

        # ✅ shared blackboard switch + bb instance
        self.use_shared_bb = bool(config.get("use_shared_bb", False))
        self.bb = get_blackboard()

        # ✅ generation params (used in _execute_subtask)
        self.max_tokens = int(config.get("max_tokens", 512))
        self.temperature = float(config.get("temperature", 0.2))
        self.top_p = float(config.get("top_p", 0.9))

        bm = config.get("base_model")
        if bm == "test":
            self.base_model = None
        elif isinstance(bm, str) and _is_openrouter_spec(bm):
            api_base, model = _parse_openrouter_spec(bm)
            api_key = os.getenv('OPENROUTER_API_KEY', os.getenv('OPENAI_API_KEY', 'EMPTY'))
            extra_headers = {
                "HTTP-Referer": "https://symphony2.0",
                "X-Title": "Symphony2.0"
            }
            self.base_model = OpenAICompatModel(api_base, model, self.system_prompt, 
                                                 api_key=api_key, extra_headers=extra_headers)
        elif isinstance(bm, str) and _is_openai_spec(bm):
            api_base, model = _parse_openai_spec(bm)
            self.base_model = OpenAICompatModel(api_base, model, self.system_prompt)
        elif BaseModel is not None and bm:
            self.base_model = BaseModel(bm, self.system_prompt, device=f"cuda:{self.gpu_id}")
        else:
            self.base_model = None

        self.memory = LocalMemory()
        self.capability_manager = CapabilityManager(self.capabilities)

        self._init_dynamic_state(
            max_inflight=config.get("max_inflight", 1),
            latency_ema_init_ms=config.get("latency_ema_init_ms", 500.0),
            latency_ema_beta=config.get("latency_ema_beta", 0.2),
        )

        if "neighbours" in config and NetworkAdapter is not None and ISEPClient is not None:
            self.network = NetworkAdapter(self.agent_id, config)
            # ✅ IMPORTANT: pass shared flags + same blackboard into ISEP
            self.isep_client = ISEPClient(
                self.agent_id,
                self.network,
                response_timeout=config.get("response_timeout", 1),
                use_shared_bb=self.use_shared_bb,
                blackboard=self.bb,
            )
            self._initialize_neighbors(config["neighbours"])
        else:
            self.network = None
            self.isep_client = None

    def _init_from_params(
        self,
        node_id: str,
        capabilities: List[str],
        system_prompt: str,
        base_model: Any,
        gpu_id: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if node_id is None:
            raise ValueError("node_id is required")

        config = config or {}

        # (optional) merge base_model.config
        try:
            bm_cfg = getattr(base_model, "config", None)
            if bm_cfg is not None:
                if not isinstance(bm_cfg, dict):
                    try:
                        bm_cfg = vars(bm_cfg)
                    except Exception:
                        bm_cfg = {}
                for k, v in (bm_cfg or {}).items():
                    config.setdefault(k, v)
        except Exception:
            pass

        self.agent_id = node_id
        self.gpu_id = gpu_id
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.capabilities = capabilities or []
        self._linucb_alpha = float(config.get("linucb_alpha", 1.0))

        self.max_tokens = int(config.get("max_tokens", 512))
        self.temperature = float(config.get("temperature", 0.2))
        self.top_p = float(config.get("top_p", 0.9))

        # ✅ keep consistent with config path
        self.use_shared_bb = bool(config.get("use_shared_bb", False))
        self.bb = get_blackboard()

        if base_model and base_model != "test":
            if _is_openrouter_spec(base_model):
                api_base, model = _parse_openrouter_spec(base_model)
                api_key = os.getenv('OPENROUTER_API_KEY', os.getenv('OPENAI_API_KEY', 'EMPTY'))
                extra_headers = {
                    "HTTP-Referer": "https://symphony2.0",
                    "X-Title": "Symphony2.0"
                }
                self.base_model = OpenAICompatModel(api_base, model, self.system_prompt,
                                                     api_key=api_key, extra_headers=extra_headers)
            elif _is_openai_spec(base_model):
                api_base, model = _parse_openai_spec(base_model)
                self.base_model = OpenAICompatModel(api_base, model, self.system_prompt)
            elif BaseModel is not None:
                self.base_model = BaseModel(base_model, self.system_prompt, device=f"cuda:{self.gpu_id}")
            else:
                self.base_model = None
        else:
            self.base_model = None

        self.memory = LocalMemory()
        self.capability_manager = CapabilityManager(self.capabilities)
        self._init_dynamic_state()

        # params-path: usually local orchestrator (no network isep)
        self.network = None
        self.isep_client = None

        # auto register (local orchestrator path)
        try:
            import symphony
            symphony.register_agent(self)
        except Exception:
            pass

    def _initialize_neighbors(self, neighbors: List[Tuple[str, str, str]]) -> None:
        for nid, host, port in neighbors:
            self.network.add_neighbor(nid, host, int(port))

    # ---------------- execution ----------------
    def _execute_subtask(self, task: Task) -> Task:
        sid = str(task["subtask_id"])
        instruction = task["steps"][sid][0]
        previous_context = " ".join(task["previous_results"])

        # ✅ BBH prompt passthrough
        if isinstance(instruction, str) and ("Final answer:" in instruction or "Task input:" in instruction or "Big-Bench Hard" in instruction):
            prompt = instruction.strip()
        else:
            prompt = self._build_task_description(instruction, previous_context)

        if self.base_model is None:
            raise RuntimeError(f"Agent {self.agent_id} base_model is None (SIMULATED). Check model loading.")

        try:
            result = self.base_model.generate(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        except TypeError:
            result = self.base_model.generate(prompt)

        task["previous_results"].append(f"{instruction} Answer: {result}")

        try:
            is_last = int(task["subtask_id"]) >= len(task["steps"])
        except Exception:
            is_last = True

        if is_last:
            task["final_result"] = result

        task["subtask_id"] = int(task.get("subtask_id", 0)) + 1
        return Task.from_dict(task)

    def execute_task(self, task):
        # accept Task or dict
        try:
            t = task.to_dict() if hasattr(task, "to_dict") else dict(task)
        except Exception:
            t = task

        steps = t.get("steps", {}) or {}
        if not isinstance(steps, dict) or len(steps) == 0:
            return ""

        try:
            t["subtask_id"] = int(t.get("subtask_id", 1))
        except Exception:
            t["subtask_id"] = 1

        while t.get("subtask_id", 1) <= len(steps):
            out = self._execute_subtask(t)
            try:
                t = out.to_dict() if hasattr(out, "to_dict") else dict(out)
            except Exception:
                t = out

        final_result = t.get("final_result", "")
        if isinstance(final_result, str):
            return final_result.strip()
        return final_result

    def get_dynamic_state(self) -> dict:
        inflight = int(getattr(self, "_inflight", 0))
        max_inflight = int(getattr(self, "_max_inflight", 1))
        latency_ms = float(getattr(self, "_latency_ema_ms", 500.0))
        reputation = float(getattr(self, "_reputation", 0.5))
        load = max(0.0, min(1.0, float(inflight) / float(max(1, max_inflight))))
        return {
            "available": inflight < max_inflight,
            "load": load,
            "latency_ms": latency_ms,
            "reputation": max(0.0, min(1.0, reputation)),
        }

    def _build_task_description(self, instruction: str, context: str) -> str:
        return (
            f"{self.system_prompt}\n"
            f"Context: {context}\n"
            f"Task: {instruction}\n"
            f"You may think step-by-step, but output ONLY the final answer.\n"
            f"The LAST line must start with exactly: Final answer:\n"
            f"Final answer:"
        )

    # ---------------- beacon handling (executor side) ----------------
    def handle_beacon(self, sender_id: str, beacon: Any) -> None:
        match_score = self.capability_manager.match(beacon["requirement"])

        load = max(0.0, min(1.0, float(self._inflight) / float(max(1, self._max_inflight))))
        available = self._inflight < self._max_inflight

        dynamic_state = {
            "load": load,
            "queue_len": int(self._inflight),
            "latency_ms": float(self._latency_ema_ms),
            "reputation": float(self._reputation),
        }
        estimate_cost = float(self._latency_ema_ms) / 1000.0

        response = BeaconResponse(
            responder_id=self.agent_id,
            task_id=beacon["task_id"],
            match_score=match_score,
            estimate_cost=estimate_cost,
            available=available,
            dynamic_state=dynamic_state,
        )

        if hasattr(response, "to_dict"):
            payload = response.to_dict()
        elif isinstance(response, dict):
            payload = response
        else:
            payload = {"_raw": str(response)}

        # ✅ shared: write to BB, do not require isep_client
        if bool(getattr(self, "use_shared_bb", False)) and getattr(self, "bb", None) is not None:
            tid = str(beacon.get("task_id", ""))
            if tid:
                payload = dict(payload) if isinstance(payload, dict) else {"_raw": str(payload)}
                # ensure keys used by collector/selector
                payload["task_id"] = tid
                payload["responder_id"] = self.agent_id
                payload.setdefault("to", sender_id)
                self.bb.append_event(tid, "beacon_response", payload, cot_id="global", step_id="0")
            return

        # legacy: require isep_client
        if self.isep_client is None:
            return
        self.isep_client.send_response(sender_id, "beacon_response", payload)

    # ---------------- Dynamic Beacon Selection (requester side) ----------------
    def assign_task(self, task: Task, topL: int = 5) -> None:
        if self.isep_client is None:
            raise RuntimeError("isep_client not initialized; cannot assign_task in local mode.")

        # ---- robust: tid ----
        if hasattr(task, "task_id"):
            tid = getattr(task, "task_id") or "task"
        else:
            tid = task.get("task_id", "task")

        if not tid:
            tid = "task"

        # ---- robust: sub_id (fixes getattr(..., task.get(...)) bug) ----
        try:
            sub_id = int(getattr(task, "subtask_id"))
        except Exception:
            sub_id = int(task.get("subtask_id", 0))

        if sub_id <= 0:
            sub_id = 1

        task_key = f"{tid}:{sub_id}"  # dispatch_id / task_key

        # ---- robust: context ----
        try:
            ctx = getattr(task, "context") or {}
        except Exception:
            ctx = task.get("context", {}) or {}

        ctx["dispatch_id"] = task_key
        ctx.setdefault("origin_task_id", tid)

        try:
            task.context = ctx
        except Exception:
            task["context"] = ctx

        # ✅ CRITICAL: enforce task_id = task_key so by-ref namespace + results match pending key
        try:
            task.task_id = task_key
        except Exception:
            task["task_id"] = task_key

        # ---- robust: steps/requirement ----
        try:
            steps = getattr(task, "steps")
        except Exception:
            steps = task.get("steps", {})

        requirement = steps[str(sub_id)][1]

        beacon = Beacon(sender=self.agent_id, task_id=task_key, requirement=requirement, ttl=2)
        responses = self.isep_client.broadcast_and_collect(beacon)

        # candidate filter: availability + threshold + non-empty id
        cand: List[Dict[str, Any]] = []
        for r in responses:
            rid = str(r.get("responder_id") or r.get("sender_id") or "").strip()
            if not rid:
                continue
            ms = float(r.get("match_score", 0.0))
            av = bool(r.get("available", True))
            if ms < 0.3 or (not av):
                continue
            cand.append(r)

        if not cand:
            raise RuntimeError(f"No available candidates for requirement={requirement}")

        # Stage-A: Top-L by match_score
        cand.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)
        cand = cand[: max(1, int(topL))]

        # Stage-B: LinUCB
        candidates_x: List[Tuple[str, List[float]]] = []
        x_map: Dict[str, List[float]] = {}

        for r in cand:
            rid = str(r.get("responder_id") or r.get("sender_id") or "").strip()
            if not rid:
                continue
            x = build_x(
                match_score=float(r.get("match_score", 0.0)),
                dynamic_state=r.get("dynamic_state", {}) or {},
                available=bool(r.get("available", True)),
                latency_scale_ms=2000.0,
            )
            candidates_x.append((rid, x))
            x_map[rid] = x

        if not candidates_x:
            raise RuntimeError("No valid candidates_x (missing responder_id/sender_id in beacon responses).")

        chosen_id = self._selector.select(candidates_x)
        if not chosen_id:
            raise RuntimeError("chosen_id empty; check beacon_response responder_id fields.")

        # Save pending for online update when result arrives
        self._pending[task_key] = {"x": x_map[chosen_id], "t0": time.time(), "executor": chosen_id}

        # ✅ delegate (by-ref if shared enabled)
        self.isep_client.delegate_task(
            chosen_id,
            task,
            send_by_ref=bool(getattr(self, "use_shared_bb", False)),
        )

    # ---------------- bandit update (requester side) ----------------
    def on_task_result(self, result: Dict[str, Any]) -> None:
        task_key = str(result.get("task_id", ""))
        if not task_key or task_key not in self._pending:
            return

        rec = self._pending.pop(task_key)
        x = rec["x"]
        t0 = float(rec["t0"])
        latency_ms = (time.time() - t0) * 1000.0

        text = str(result.get("result", ""))
        ok = (text.strip() != "") and (not text.startswith("[ERROR]")) and (not text.startswith("[AGENT_ERROR]"))

        lat_norm = min(1.0, latency_ms / 2000.0)
        reward = (1.0 if ok else 0.0) - 0.2 * (lat_norm ** 0.5)
        reward = max(0.0, min(1.0, reward))

        self._selector.update(x, reward)
        self._reputation = max(0.0, min(1.0, 0.95 * self._reputation + 0.05 * (1.0 if ok else 0.0)))

    # ---------------- runtime helper loop ----------------
    def process_incoming_once(self, timeout: float = 0.05) -> None:
        """
        executor:
          - receive beacon -> handle_beacon
          - receive task -> execute -> submit_result
        requester:
          - shared: poll blackboard kv for results -> update bandit
          - legacy: receive_result -> update bandit
        """
        if self.isep_client is None:
            return

        # 1) beacons
        sid, _, beacon = self.isep_client.receive_beacon(timeout=timeout)
        if beacon is not None:
            self.handle_beacon(sid, beacon)

        # 2) tasks (executor side)
        sid, _, task = self.isep_client.receive_task(timeout=timeout)
        if task is not None:
            task_obj = Task.from_dict(task) if isinstance(task, dict) else task

            # robust ctx/dispatch_id
            try:
                ctx = task_obj.get("context", {}) or {}
            except Exception:
                ctx = getattr(task_obj, "context", {}) or {}
            dispatch_id = str(ctx.get("dispatch_id", "") or (task_obj.get("task_id", "") if hasattr(task_obj, "get") else ""))

            done_text = self.execute_task(task_obj)

            # ✅ shared: write to BB; legacy: also send over network
            self.isep_client.submit_result(
                target_id=sid,
                result=done_text,
                previous_results=[],
                task_id=dispatch_id,
                also_send_network=(not bool(getattr(self, "use_shared_bb", False))),
            )

        # 3) results (requester side)
        if bool(getattr(self, "use_shared_bb", False)) and getattr(self, "bb", None) is not None:
            for task_key in list(self._pending.keys()):
                rec = self._pending.get(task_key) or {}
                executor = str(rec.get("executor", "")).strip()
                if not executor:
                    continue
                res = self.bb.kv_get(task_key, f"result:{executor}", default=None)
                if res:
                    if isinstance(res, dict):
                        res.setdefault("task_id", task_key)
                    else:
                        res = {"task_id": task_key, "result": str(res)}
                    self.on_task_result(res)
            return

        # legacy network results
        sid, _, res = self.isep_client.receive_result(timeout=timeout)
        if res is not None:
            self.on_task_result(res)
