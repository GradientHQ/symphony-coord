# core/embeddings.py
"""
✅ Symphony 2.0: Embedding computation and similarity matching.

Responsibilities:
- Text embedding via OpenAI / OpenRouter / local models
- Agent embedding (cached, static)
- Task embedding (cached by content)
- Cosine similarity computation

This module isolates embedding implementation details from routing logic.
"""

from __future__ import annotations

import os
import math
import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Any


__all__ = [
    "EmbeddingProvider",
    "get_text_embedding",
    "cosine_similarity",
    "embed_subtask",
    "embed_agent",
    "sim_emb",
]


# -------------------------
# Embedding provider (abstract interface)
# -------------------------

class EmbeddingProvider:
    """Abstract embedding provider interface."""
    
    def embed(self, text: str) -> List[float]:
        """Compute embedding for text. Returns empty list on error."""
        raise NotImplementedError
    
    def cosine_sim(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        
        # ✅ P1: Map cosine from [-1,1] to [0,1] for better discrimination
        cos_val = dot / (norm_a * norm_b)
        return 0.5 * (cos_val + 1.0)  # Maps [-1,1] -> [0,1]


class DummyEmbeddingProvider(EmbeddingProvider):
    """
    ✅ P0: Dummy provider that returns empty embeddings (no-op fallback).
    Prevents NotImplementedError when no provider is available.
    """
    
    def embed(self, text: str) -> List[float]:
        """Return empty embedding (signals embedding unavailable)."""
        return []


# -------------------------
# OpenAI/OpenRouter embedding provider
# -------------------------

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    ✅ OpenAI/OpenRouter embedding API provider.
    
    P0: Automatically detects OpenRouter vs OpenAI based on API key.
    Uses correct endpoint and headers for each service.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        
        # ✅ P0: Detect OpenRouter vs OpenAI by API key prefix
        openai_key = os.getenv("OPENAI_API_KEY", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        
        if api_key:
            self.api_key = api_key
        elif openai_key:
            self.api_key = openai_key
        elif openrouter_key:
            self.api_key = openrouter_key
        else:
            self.api_key = ""
        
        # ✅ P0: Set correct base_url based on which key is used
        if base_url:
            self.base_url = base_url
        elif openrouter_key and not openai_key:
            # OpenRouter embeddings endpoint
            self.base_url = "https://openrouter.ai/api/v1/embeddings"
        else:
            # OpenAI embeddings endpoint
            self.base_url = "https://api.openai.com/v1/embeddings"
        
        self._is_openrouter = bool(openrouter_key and not openai_key)
    
    def embed(self, text: str) -> List[float]:
        """Compute embedding via OpenAI/OpenRouter API."""
        if not text or not text.strip() or not self.api_key:
            return []
        
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # ✅ P0: Add OpenRouter-specific headers if using OpenRouter
            if self._is_openrouter:
                headers["HTTP-Referer"] = "https://symphony2.0"
                headers["X-Title"] = "Symphony2.0"
            
            payload = {
                "model": self.model,
                "input": text.strip(),
            }
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data["data"][0]["embedding"]
            else:
                # ✅ A: Log API error (first 200 chars only, avoid sensitive info)
                error_preview = resp.text[:200] if hasattr(resp, "text") else ""
                print(f"[EMBEDDING_API_ERROR] Status {resp.status_code}, preview: {error_preview}")
        except Exception as e:
            # Silent fallback for network errors
            pass
        return []


# -------------------------
# Local sentence-transformers provider
# -------------------------

class LocalEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy load local model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise RuntimeError(
                    f"LocalEmbeddingProvider requires 'sentence-transformers'. "
                    f"Install via: pip install sentence-transformers"
                )
        return self._model
    
    def embed(self, text: str) -> List[float]:
        """Compute embedding via local model."""
        if not text or not text.strip():
            return []
        
        try:
            model = self._load_model()
            emb = model.encode(text.strip(), convert_to_numpy=True)
            return emb.tolist()
        except Exception:
            return []


# -------------------------
# Global provider (singleton, lazy init)
# -------------------------

_global_provider: Optional[EmbeddingProvider] = None
_embedding_dim_logged: bool = False  # ✅ Track if dimension has been logged


def _get_provider() -> EmbeddingProvider:
    """Get or create global embedding provider (with one-time logging)."""
    global _global_provider
    if _global_provider is None:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        
        # Try OpenAI/OpenRouter first (if API key available)
        if openai_key or openrouter_key:
            _global_provider = OpenAIEmbeddingProvider()
            provider_type = "OpenRouter" if (openrouter_key and not openai_key) else "OpenAI"
            # ✅ 1: One-time logging for provider selection (ensure reproducibility)
            print(f"[EMBEDDING_PROVIDER] Selected: {provider_type}, model: {_global_provider.model}, "
                  f"endpoint: {_global_provider.base_url}")
        else:
            # Fallback to local (if available)
            try:
                _global_provider = LocalEmbeddingProvider()
                print(f"[EMBEDDING_PROVIDER] Selected: Local, model: {_global_provider.model_name}")
            except RuntimeError:
                # ✅ P0: Use DummyEmbeddingProvider instead of abstract class
                _global_provider = DummyEmbeddingProvider()
                print("[EMBEDDING_PROVIDER] Selected: Dummy (no embeddings available)")
        
        # ✅ 1: Log embedding dimension after first successful computation (optional check)
        # This will be logged when first embedding is computed
    
    return _global_provider


# -------------------------
# Public API: embedding computation
# -------------------------

# Cache for embeddings (LRU by content hash)
_embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
_embedding_cache_maxsize = 1000


def get_text_embedding(text: str, provider: Optional[EmbeddingProvider] = None) -> List[float]:
    """
    ✅ Compute text embedding (LRU cached by content hash).
    
    Args:
        text: Input text
        provider: EmbeddingProvider instance (if None, uses global)
    
    Returns:
        Embedding vector (empty list on error)
    """
    if not text or not text.strip():
        return []
    
    provider = provider or _get_provider()
    
    # ✅ P0: Use SHA256 hash as cache key (avoid collisions from prefix truncation)
    text_normalized = text.strip()
    text_key = hashlib.sha256(text_normalized.encode("utf-8")).hexdigest()
    
    # ✅ A: LRU cache lookup (move to end on hit)
    if text_key in _embedding_cache:
        _embedding_cache.move_to_end(text_key)
        return _embedding_cache[text_key]
    
    # Compute
    emb = provider.embed(text)
    
    # ✅ A: LRU cache update (evict oldest if at capacity)
    if emb:
        if len(_embedding_cache) >= _embedding_cache_maxsize:
            _embedding_cache.popitem(last=False)  # Remove oldest
        _embedding_cache[text_key] = emb
        _embedding_cache.move_to_end(text_key)  # Mark as recently used
        
        # ✅ 1: Log embedding dimension after first successful computation
        global _embedding_dim_logged
        if not _embedding_dim_logged and len(emb) > 0:
            print(f"[EMBEDDING_DIM] Vector dimension: {len(emb)}")
            _embedding_dim_logged = True
    
    return emb


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    ✅ Compute cosine similarity between two vectors (mapped to [0, 1]).
    
    Maps standard cosine [-1, 1] to [0, 1] for better discrimination:
    sim = 0.5 * (cos + 1.0)
    
    Args:
        vec_a: First vector
        vec_b: Second vector
    
    Returns:
        Similarity [0, 1] (clipped for numerical stability)
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    # ✅ P1: Map cosine from [-1,1] to [0,1], then clip for numerical stability
    cos_val = dot / (norm_a * norm_b)
    sim = 0.5 * (cos_val + 1.0)  # Maps [-1,1] -> [0,1]
    return max(0.0, min(1.0, sim))  # ✅ C: Clip for numerical stability


# -------------------------
# Agent capability text extraction
# -------------------------

def _get_agent_capability_text(agent: Any) -> str:
    """
    Build capability description text from agent.
    
    Priority:
    1. sys_prompt (if contains [STRENGTHS] or capability descriptions)
    2. Capability tags concatenated with descriptions
    3. Fallback: capability tags only
    """
    parts: List[str] = []
    
    # Try sys_prompt first
    sys_prompt = getattr(agent, "system_prompt", None) or getattr(agent, "sys_prompt", None)
    if isinstance(sys_prompt, str) and sys_prompt.strip():
        sp = sys_prompt.strip()
        if "[STRENGTHS]" in sp:
            idx = sp.find("[STRENGTHS]")
            end = sp.find("\n[", idx + 1)
            if end > idx:
                strengths = sp[idx:end].strip()
                parts.append(strengths)
        elif "[ROLE]" in sp or "capabilities" in sp.lower():
            parts.append(sp[:300])
    
    # Add capability tags
    caps = []
    if hasattr(agent, "capabilities"):
        caps = agent.capabilities if isinstance(agent.capabilities, list) else []
    elif hasattr(agent, "capability_manager"):
        try:
            caps = agent.capability_manager.list_capabilities()
        except Exception:
            pass
    
    if caps:
        caps_str = ", ".join(str(c) for c in caps[:10])
        parts.append(f"Capabilities: {caps_str}")
    
    # Fallback
    if not parts:
        aid = str(getattr(agent, "agent_id", ""))
        parts.append(f"Agent: {aid}")
    
    return " ".join(parts).strip()


# -------------------------
# Agent embedding (cached by agent_id)
# -------------------------

_agent_embedding_cache: Dict[str, List[float]] = {}


def embed_agent(agent: Any, provider: Optional[EmbeddingProvider] = None) -> List[float]:
    """
    ✅ Get or compute agent embedding (cached by agent_id).
    
    Args:
        agent: Agent object
        provider: EmbeddingProvider instance (if None, uses global)
    
    Returns:
        Agent embedding vector (empty list on error)
    """
    # ✅ P0: agent_id fallback chain: agent_id -> node_id -> name -> id
    aid = (
        str(getattr(agent, "agent_id", "")) or
        str(getattr(agent, "node_id", "")) or
        str(getattr(agent, "name", "")) or
        str(getattr(agent, "id", "")) or
        ""
    ).strip()
    
    if not aid:
        return []
    
    # Check cache
    if aid in _agent_embedding_cache:
        return _agent_embedding_cache[aid]
    
    # Build capability text and compute embedding
    cap_text = _get_agent_capability_text(agent)
    emb = get_text_embedding(cap_text, provider)
    
    # Cache
    if emb:
        _agent_embedding_cache[aid] = emb
    
    return emb


# -------------------------
# Subtask embedding (cached by content)
# -------------------------

def embed_subtask(subtask: Dict[str, Any], provider: Optional[EmbeddingProvider] = None) -> List[float]:
    """
    ✅ Get or compute subtask embedding (cached by content).
    
    Args:
        subtask: Subtask dict with "input", "description", "requirement", "task_id", "prompt" fields
        provider: EmbeddingProvider instance (if None, uses global)
    
    Returns:
        Subtask embedding vector (empty list on error)
    """
    # Build task text (requirement + subtask text if available)
    req = str(subtask.get("requirement", "")).strip()
    task_text = str(subtask.get("input", "") or subtask.get("description", "") or "").strip()
    
    # ✅ 4: If subtask has no input/description, try to add task_id or prompt summary
    if not task_text:
        # Try task_id for uniqueness
        task_id = str(subtask.get("task_id", "") or subtask.get("id", "")).strip()
        if task_id:
            task_text = f"{req} [Task: {task_id}]".strip()
        else:
            # Try raw_data.prompt if available (for humaneval/gsm8k)
            raw_data = subtask.get("raw_data", {})
            if isinstance(raw_data, dict):
                prompt = raw_data.get("prompt", "")
                if prompt:
                    task_text = f"{req} {prompt[:200]}".strip()  # First 200 chars
    
    if req and req not in task_text:
        task_text = f"{req} {task_text}".strip()
    
    if not task_text:
        return []
    
    return get_text_embedding(task_text, provider)


# -------------------------
# Similarity computation
# -------------------------

def sim_emb(agent: Any, subtask: Dict[str, Any], provider: Optional[EmbeddingProvider] = None) -> float:
    """
    ✅ Compute embedding-based similarity between agent and subtask.
    
    Args:
        agent: Agent object
        subtask: Subtask dict
        provider: EmbeddingProvider instance (if None, uses global)
    
    Returns:
        Cosine similarity [0, 1] (fallback: 0.5 if embedding fails)
    """
    agent_emb = embed_agent(agent, provider)
    if not agent_emb:
        return 0.5  # Fallback: neutral similarity
    
    subtask_emb = embed_subtask(subtask, provider)
    if not subtask_emb:
        return 0.5  # Fallback: neutral similarity
    
    return cosine_similarity(agent_emb, subtask_emb)

