"""One-shot model warm-up to reduce first-item LLM latency (PRD §5e G3).

Groq is stateless HTTP — no warm-up needed. Ollama loads models lazily on first
request, so a throwaway prompt here prevents cold-start latency on the first item.
"""

from __future__ import annotations

from loguru import logger

from app.core.config import settings
from app.core.llm import get_chat_model

_warmed: bool = False


async def warmup_llm() -> None:
    """Warm the model if using Ollama; no-op for Groq. Idempotent."""
    global _warmed
    if _warmed:
        return
    if settings.llm_provider != "ollama":
        logger.info("llm_warmup_skipped", provider=settings.llm_provider)
        _warmed = True
        return
    logger.info("llm_warmup_start", provider=settings.llm_provider)
    try:
        await get_chat_model(role="quality").ainvoke("ok")
        _warmed = True
        logger.info("llm_warmup_done")
    except Exception as exc:
        # Non-fatal: warmup failure just means slower first item.
        logger.warning("llm_warmup_failed", error=str(exc))
