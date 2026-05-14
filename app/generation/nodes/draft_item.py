"""draft_item node — calls the LLM with structured output for the current template item."""

from __future__ import annotations

import asyncio
import json

import httpx
from loguru import logger
from pydantic import ValidationError

from app.core.llm import get_chat_model
from app.core.ratelimit import groq_limiter
from app.core.config import settings
from app.generation.prompts import load_prompt
from app.generation.state import ChecklistState, DraftChecklistItem
from app.retrieval.hybrid_search import SearchHit

_PROMPT_TEMPLATE = load_prompt("draft_item", version="v1")

_STRICT_SUFFIX = (
    "\n\nIMPORTANT: Your previous response did not conform to the JSON schema. "
    "Return ONLY valid JSON matching the schema. No prose, no markdown fences."
)

_LLM_TIMEOUT = 60  # seconds per G2


def _build_evidence_blocks(hits: list[SearchHit]) -> str:
    parts: list[str] = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[E{i} doc={hit.doc_id} page={hit.page_number}]\n{hit.context_text}")
    return "\n\n".join(parts) if parts else "(No evidence retrieved)"


def _coerce_unclear(rationale: str) -> DraftChecklistItem:
    return DraftChecklistItem(
        status="unclear",
        confidence=0.0,
        rationale=rationale,
        cited_evidence=[],
    )


async def _call_llm(prompt_text: str, use_strict: bool = False) -> DraftChecklistItem:
    """Single LLM call with timeout. Raises on failure.

    groq_limiter wraps the entire ainvoke so no slot is released before the
    request fires — prevents two concurrent calls from both passing the limiter.
    """
    full_prompt = prompt_text + (_STRICT_SUFFIX if use_strict else "")
    model = get_chat_model(role="quality").with_structured_output(DraftChecklistItem)

    if settings.llm_provider == "groq":
        async with groq_limiter:
            result = await asyncio.wait_for(model.ainvoke(full_prompt), timeout=_LLM_TIMEOUT)
    else:
        result = await asyncio.wait_for(model.ainvoke(full_prompt), timeout=_LLM_TIMEOUT)

    if not isinstance(result, DraftChecklistItem):
        raise ValueError(f"Unexpected output type: {type(result)}")
    return result


async def draft_item(state: ChecklistState) -> dict[str, object]:
    """Draft one ChecklistItem via structured LLM call with one retry on validation failure."""
    template = state["template"]
    slug = state["current_item_slug"]
    assert slug is not None

    item = next(i for i in template.items if i.slug == slug)
    hits = (state.get("search_hits_by_item") or {}).get(slug, [])

    evidence_blocks = _build_evidence_blocks(hits)
    schema_json = json.dumps(DraftChecklistItem.model_json_schema(), indent=2)

    prompt_text = _PROMPT_TEMPLATE.format(
        schema=schema_json,
        item_title=item.title,
        item_description=item.description,
        item_category=item.category,
        item_required=str(item.required),
        learned_patterns_json="[]",
        few_shot_pairs_json="[]",
        evidence_blocks=evidence_blocks,
    )

    model_version = (
        settings.ollama_model_quality
        if settings.llm_provider == "ollama"
        else settings.groq_model_quality
    )

    # Recoverable LLM failures: validation error (bad JSON), timeout, provider HTTP errors.
    # Infrastructure errors (ImportError, AttributeError, etc.) propagate and fail the run.
    _RECOVERABLE = (ValidationError, asyncio.TimeoutError, httpx.HTTPError, ValueError)

    draft: DraftChecklistItem
    try:
        draft = await _call_llm(prompt_text, use_strict=False)
    except _RECOVERABLE as first_exc:
        first_reason = (
            "LLM call timed out" if isinstance(first_exc, asyncio.TimeoutError)
            else str(first_exc)[:120]
        )
        logger.bind(item_slug=slug, error=first_reason).warning("draft_item_first_attempt_failed")
        # One retry with strict prompt suffix
        try:
            draft = await _call_llm(prompt_text, use_strict=True)
        except _RECOVERABLE as second_exc:
            second_reason = (
                "LLM call timed out"
                if isinstance(second_exc, asyncio.TimeoutError)
                else "LLM failed to produce valid item structure"
            )
            logger.bind(item_slug=slug, error=str(second_exc)[:120]).warning(
                "draft_item_second_attempt_failed"
            )
            draft = _coerce_unclear(second_reason)

    in_progress = dict(state.get("items_in_progress") or {})
    # Store as a sentinel dict carrying both draft and template item so validate_item can read it.
    # The final ChecklistItem is written by validate_item.
    in_progress[slug] = {"_draft": draft, "_template_item": item}  # type: ignore[assignment]

    return {
        "items_in_progress": in_progress,
        "model_version": model_version,
    }
