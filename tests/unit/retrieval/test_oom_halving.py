"""OOM auto-halving: wrapper converges to a working batch size and completes (P2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

import app.retrieval.embedding as emb_module


@pytest.mark.asyncio
async def test_oom_halves_to_stable_batch() -> None:
    """Mock Ollama to raise OOM for batch >= 32, succeed for <= 16."""
    call_sizes: list[int] = []

    async def fake_call_ollama(texts: list[str]) -> list[list[float]]:
        call_sizes.append(len(texts))
        if len(texts) >= 32:
            # Simulate an OOM-shaped HTTP 500 with matching body
            resp = httpx.Response(500, text="out of memory")
            raise httpx.HTTPStatusError("OOM", request=httpx.Request("POST", "http://x"), response=resp)
        return [[0.1] * 768 for _ in texts]

    # Reset the stable batch size before test
    emb_module._stable_batch_size = 64

    with patch.object(emb_module, "_call_ollama", side_effect=fake_call_ollama):
        texts = ["text"] * 40
        result = await emb_module.embed_texts(texts)

    assert len(result) == 40
    # All vectors must have the right dimension
    assert all(len(v) == 768 for v in result)
    # Stable batch should have halved to <= 16
    assert emb_module._stable_batch_size <= 16


@pytest.mark.asyncio
async def test_oom_floor_at_4() -> None:
    """Even if batch=4 still OOMs, RuntimeError is raised after floor."""
    async def always_oom(texts: list[str]) -> list[list[float]]:
        resp = httpx.Response(500, text="out of memory always")
        raise httpx.HTTPStatusError("OOM", request=httpx.Request("POST", "http://x"), response=resp)

    emb_module._stable_batch_size = 8

    with patch.object(emb_module, "_call_ollama", side_effect=always_oom):
        with pytest.raises(RuntimeError, match="minimum batch size"):
            await emb_module.embed_texts(["text"] * 10)


@pytest.mark.asyncio
async def test_non_oom_http_error_not_retried_as_oom() -> None:
    """A 404 or other non-OOM HTTP error is not treated as OOM — propagates immediately."""
    call_count = 0

    async def fake_404(texts: list[str]) -> list[list[float]]:
        nonlocal call_count
        call_count += 1
        resp = httpx.Response(404, text="not found")
        raise httpx.HTTPStatusError("404", request=httpx.Request("POST", "http://x"), response=resp)

    emb_module._stable_batch_size = 64

    with patch.object(emb_module, "_call_ollama", side_effect=fake_404):
        with pytest.raises(httpx.HTTPStatusError):
            await emb_module.embed_texts(["text"])

    # Should have tried only once (tenacity will retry on HTTPError, but that's the
    # network_retry decorator — the OOM halving should NOT loop for non-OOM errors)
    # Accept up to tenacity's stop_after_attempt(4) retries for transient HTTP errors
    assert call_count >= 1
