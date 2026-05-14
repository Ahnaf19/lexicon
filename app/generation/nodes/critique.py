"""critique node — apply promoted learned patterns to the current ChecklistItem (PRD §5f Layer 3).

L3 gate: if no actionable patterns apply to this item, return {} with zero LLM calls.
Only rename_rule, status_default, style_preference, and category_remap are applied here;
template_addition/removal are handled earlier in load_template.
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx
from loguru import logger
from pydantic import ValidationError

from app.core.config import settings
from app.core.llm import get_chat_model
from app.core.ratelimit import groq_limiter
from app.generation.prompts import load_prompt
from app.generation.state import ChecklistState
from app.models.pydantic_models import (
    CategoryRemap,
    ChecklistItem,
    CritiqueResult,
    LearnedPattern,
    RenameRule,
    StatusDefault,
    StylePreference,
)

_PROMPT_TEMPLATE = load_prompt("critique", version="v1")
_LLM_TIMEOUT = settings.llm_critique_timeout_s
_RECOVERABLE = (ValidationError, asyncio.TimeoutError, httpx.HTTPError, ValueError)

_ACTIONABLE_TYPES = frozenset(
    {"rename_rule", "status_default", "style_preference", "category_remap"}
)


def _match_rename_rule(rule: RenameRule, item: ChecklistItem) -> bool:
    if rule.scope_category and rule.scope_category.lower() != str(item.category).lower():
        return False
    return rule.from_text.lower() in item.title.lower()


def _match_status_default(rule: StatusDefault, item: ChecklistItem) -> bool:
    # UUID endswith a slug is never true; fall through to the title-based check.
    if item.source_template_item_id is not None and str(item.source_template_item_id).endswith(rule.item_slug):
        return True
    return rule.item_slug.lower() in item.title.lower().replace(" ", "_")


def _match_category_remap(rule: CategoryRemap, item: ChecklistItem) -> bool:
    try:
        return bool(re.search(rule.matches_title_regex, item.title, re.IGNORECASE))
    except re.error:
        return False


def _find_matching_patterns(
    item: ChecklistItem,
    patterns: list[LearnedPattern],
) -> list[LearnedPattern]:
    """Return patterns whose rules actually apply to this item (L3 gate)."""
    matched: list[LearnedPattern] = []
    for pattern in patterns:
        if pattern.pattern_type not in _ACTIONABLE_TYPES:
            continue
        try:
            if pattern.pattern_type == "rename_rule":
                rule = RenameRule.model_validate(pattern.rule_json)
                if _match_rename_rule(rule, item):
                    matched.append(pattern)
            elif pattern.pattern_type == "status_default":
                rule = StatusDefault.model_validate(pattern.rule_json)
                if _match_status_default(rule, item):
                    matched.append(pattern)
            elif pattern.pattern_type == "style_preference":
                # style_preference is a global phrasing guideline — always applies.
                matched.append(pattern)
            elif pattern.pattern_type == "category_remap":
                rule = CategoryRemap.model_validate(pattern.rule_json)
                if _match_category_remap(rule, item):
                    matched.append(pattern)
        except Exception as exc:
            logger.bind(
                pattern_id=str(pattern.id), error=str(exc)[:80]
            ).warning("critique_rule_parse_failed")
    return matched


async def critique(state: ChecklistState) -> dict[str, object]:
    """Apply promoted patterns to the current item; gated — no LLM call if no rules match."""
    slug = state.get("current_item_slug")
    if slug is None:
        return {}

    in_progress = dict(state.get("items_in_progress") or {})
    item = in_progress.get(slug)

    if not isinstance(item, ChecklistItem):
        return {}

    learned_patterns: list[LearnedPattern] = list(state.get("learned_patterns") or [])
    matched = _find_matching_patterns(item, learned_patterns)

    if not matched:
        # L3 gate: no rules apply — return unchanged, zero LLM calls.
        return {}

    matched_rules_json = json.dumps(
        [{"pattern_type": p.pattern_type, "rule_json": p.rule_json} for p in matched],
        default=str,
    )
    # CritiqueResult excludes evidence/confidence/id — the LLM never sees them in the
    # schema, so it can't accidentally omit evidence and trigger evidence_required_when_present.
    schema_json = json.dumps(CritiqueResult.model_json_schema(), indent=2)
    item_json = item.model_dump_json()

    # Use .replace() not .format() — JSON braces in schema break .format() (L7).
    prompt_text = (
        _PROMPT_TEMPLATE
        .replace("{schema}", schema_json)
        .replace("{matched_rules_json}", matched_rules_json)
        .replace("{item_json}", item_json)
    )

    model = get_chat_model(role="quality").with_structured_output(CritiqueResult)

    try:
        if settings.llm_provider == "groq":
            async with groq_limiter:
                result = await asyncio.wait_for(
                    model.ainvoke(prompt_text), timeout=_LLM_TIMEOUT
                )
        else:
            result = await asyncio.wait_for(model.ainvoke(prompt_text), timeout=_LLM_TIMEOUT)

        if not isinstance(result, CritiqueResult):
            raise ValueError(f"Unexpected output: {type(result)}")

    except _RECOVERABLE as exc:
        reason = "LLM call timed out" if isinstance(exc, asyncio.TimeoutError) else (str(exc) or repr(exc))[:120]
        logger.bind(item_slug=slug, error=reason).warning("critique_llm_failed_fallback")
        # Never fail the pipeline — fall back to un-critiqued item.
        return {}

    # Merge mutable LLM-rewritten fields onto the original item; restore all immutable fields.
    corrected = item.model_copy(
        update={
            "title": result.title,
            "description": result.description,
            "rationale": result.rationale,
            "category": result.category,
            "status": result.status,
            "required": result.required,
            "evidence": item.evidence,
            "confidence": item.confidence,
            "id": item.id,
            "source_template_item_id": item.source_template_item_id,
            "learned_from_pattern_ids": [p.id for p in matched],
        }
    )

    logger.bind(
        item_slug=slug,
        patterns_applied=[str(p.id) for p in matched],
    ).info("critique_applied")

    in_progress[slug] = corrected
    return {"items_in_progress": in_progress}
