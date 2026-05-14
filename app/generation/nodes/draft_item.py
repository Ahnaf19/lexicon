"""draft_item node — calls the LLM with structured output for the current template item."""

from __future__ import annotations

import asyncio
import json

import httpx
from loguru import logger
from pydantic import ValidationError

from app.core.config import settings
from app.core.db import SessionLocal
from app.core.llm import get_chat_model
from app.core.ratelimit import groq_limiter
from app.generation.prompts import load_prompt
from app.generation.state import ChecklistState, DraftChecklistItem
from app.learning.few_shot_bank import retrieve_few_shot
from app.models.pydantic_models import ChecklistItem, LearnedPattern
from app.retrieval.hybrid_search import SearchHit

_PROMPT_TEMPLATE = load_prompt("draft_item", version="v1")

_STRICT_SUFFIX = (
    "\n\nIMPORTANT: Your previous response did not conform to the JSON schema. "
    "Return ONLY valid JSON matching the schema. No prose, no markdown fences."
)

_LLM_TIMEOUT = settings.llm_draft_timeout_s


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


def _render_few_shot_pairs(pairs: list[tuple[ChecklistItem, ChecklistItem]]) -> str:
    """Serialise (original_draft, final_item) pairs for prompt injection."""
    out = []
    for original, final in pairs:
        out.append(
            {
                "original_draft": original.model_dump(mode="json"),
                "final_item": final.model_dump(mode="json"),
            }
        )
    return json.dumps(out, default=str)


def _render_draft_patterns(patterns: list[LearnedPattern]) -> str:
    """Serialise patterns relevant at draft time (style_preference, status_default)."""
    relevant = [
        {"pattern_type": p.pattern_type, "rule_json": p.rule_json}
        for p in patterns
        if p.pattern_type in ("style_preference", "status_default")
    ]
    return json.dumps(relevant, default=str)


async def draft_item(state: ChecklistState) -> dict[str, object]:
    """Draft one ChecklistItem via structured LLM call with one retry on validation failure."""
    template = state["template"]
    slug = state["current_item_slug"]
    assert slug is not None

    item = next(i for i in template.items if i.slug == slug)
    hits = (state.get("search_hits_by_item") or {}).get(slug, [])
    learned_patterns: list[LearnedPattern] = list(state.get("learned_patterns") or [])
    model_version = (
        settings.ollama_model_quality
        if settings.llm_provider == "ollama"
        else settings.groq_model_quality
    )

    # Fast-path: no evidence retrieved → skip LLM, validate_item enforces
    # "present requires evidence" anyway, so any LLM guess would be overridden.
    if not hits:
        in_progress = dict(state.get("items_in_progress") or {})
        in_progress[slug] = {
            "_draft": _coerce_unclear("No evidence retrieved; item presence cannot be determined."),
            "_template_item": item,
        }
        return {"items_in_progress": in_progress, "model_version": model_version}

    evidence_blocks = _build_evidence_blocks(hits)
    evidence_summary = " ".join(h.snippet for h in hits)[:500]

    # CIPHER: retrieve top-3 few-shot pairs by embedding similarity (L2).
    few_shot_pairs: list[tuple[ChecklistItem, ChecklistItem]] = []
    try:
        async with SessionLocal() as session:
            few_shot_pairs = await retrieve_few_shot(
                session=session,
                template_item=item,
                doc_type=template.doc_type,
                evidence_summary=evidence_summary,
                k=3,
            )
    except Exception as exc:
        logger.bind(item_slug=slug, error=str(exc)[:120]).warning(
            "draft_item_few_shot_retrieval_failed"
        )

    schema_json = json.dumps(DraftChecklistItem.model_json_schema(), indent=2)

    # Use .replace() not .format() — JSON braces in schema break .format() (L7).
    prompt_text = (
        _PROMPT_TEMPLATE
        .replace("{schema}", schema_json)
        .replace("{item_title}", item.title)
        .replace("{item_description}", item.description)
        .replace("{item_category}", item.category)
        .replace("{item_required}", str(item.required))
        .replace("{learned_patterns_json}", _render_draft_patterns(learned_patterns))
        .replace("{few_shot_pairs_json}", _render_few_shot_pairs(few_shot_pairs))
        .replace("{evidence_blocks}", evidence_blocks)
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
            logger.bind(item_slug=slug, error=second_reason).warning(
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
