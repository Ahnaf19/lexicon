"""One-shot qwen3 warm-up to reduce first-item LLM latency (PRD §5e G3)."""

from __future__ import annotations

from loguru import logger

from app.core.llm import get_chat_model

_warmed: bool = False


async def warmup_llm() -> None:
    """Issue a throwaway prompt to warm the model; idempotent."""
    global _warmed
    if _warmed:
        return
    logger.debug("llm_warmup_start")
    try:
        await get_chat_model(role="quality").ainvoke("ok")
        _warmed = True
        logger.debug("llm_warmup_done")
    except Exception as exc:
        # Non-fatal: warmup failure just means slower first item.
        logger.warning("llm_warmup_failed", error=str(exc))
