"""Ollama nomic-embed-text embedding wrapper (PRD §5c).

Performance constraints:
  P2 — auto-halves batch on OOM errors, floor 4
  P3 — asyncio.Semaphore(2) across concurrent batch calls
  P8 — module-level query cache keyed by query string
  P10 — warmup() primes the Metal/CUDA backend before timed batches
"""

from __future__ import annotations

import asyncio
import re

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_QUERY_CACHE: dict[str, list[float]] = {}
_warmed: bool = False
_stable_batch_size: int = 64  # updated downward on OOM; read by indexing.py for logging
_SEM = asyncio.Semaphore(2)  # max two concurrent batch requests (P3)

_OOM_RE = re.compile(r"out of memory|context length|too large", re.IGNORECASE)

# Shared httpx client — reused across calls, closed at process exit
_CLIENT: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        _CLIENT = httpx.AsyncClient(base_url=str(settings.ollama_base_url), timeout=120.0)
    return _CLIENT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def _call_ollama(texts: list[str]) -> list[list[float]]:
    """Single POST /api/embed for `texts`. Raises on HTTP/timeout; caller handles OOM."""
    resp = await _get_client().post(
        "/api/embed",
        json={"model": settings.embedding_model, "input": texts},
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embeddings": [[...], ...]}
    return data["embeddings"]


async def _embed_batch_with_halving(texts: list[str]) -> list[list[float]]:
    """Embed texts with auto-halving on OOM (P2). Updates _stable_batch_size."""
    global _stable_batch_size

    batch = _stable_batch_size
    while True:
        try:
            results: list[list[float]] = []
            for i in range(0, len(texts), batch):
                chunk = texts[i : i + batch]
                async with _SEM:
                    vecs = await _call_ollama(chunk)
                results.extend(vecs)
            return results
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            if _OOM_RE.search(body):
                if batch <= 4:
                    raise RuntimeError(
                        f"Could not embed {len(texts)} texts even at minimum batch size 4 — OOM persists"
                    ) from exc
                new_batch = max(batch // 2, 4)
                logger.bind(old_batch=batch, new_batch=new_batch).warning(
                    "embedding_batch_oom"
                )
                batch = new_batch
                _stable_batch_size = batch
                continue
            raise
        except Exception as exc:
            body = str(exc)
            if _OOM_RE.search(body):
                if batch <= 4:
                    raise RuntimeError(
                        f"Could not embed {len(texts)} texts even at minimum batch size 4 — OOM persists"
                    ) from exc
                new_batch = max(batch // 2, 4)
                logger.bind(old_batch=batch, new_batch=new_batch).warning(
                    "embedding_batch_oom"
                )
                batch = new_batch
                _stable_batch_size = batch
                continue
            raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def warmup() -> None:
    """Prime the Ollama model backend with a throwaway embed call (P10).

    Idempotent — no-op after the first successful call in this process.
    """
    global _warmed
    if _warmed:
        return
    try:
        await _call_ollama(["warmup"])
        _warmed = True
        logger.info("embedding_warmup_done")
    except Exception as exc:
        logger.bind(error=str(exc)).warning("embedding_warmup_failed")


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns parallel list of 768-dim vectors."""
    if not texts:
        return []
    return await _embed_batch_with_halving(texts)


async def embed_query(query: str) -> list[float]:
    """Embed a single query string, with result cache (P8)."""
    if query in _QUERY_CACHE:
        return _QUERY_CACHE[query]
    vecs = await embed_texts([query])
    result = vecs[0]
    _QUERY_CACHE[query] = result
    return result
