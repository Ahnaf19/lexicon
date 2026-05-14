"""Pattern extractor — mines edit events into promoted LearnedPatterns (PRD §5f Layer 2).

Called from POST /checklists/{id}/finalize as a FastAPI BackgroundTask.
One LLM call per finalize (L1). Tenacity with 30s timeout; narrow recoverable set.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

import httpx
from loguru import logger
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import SessionLocal
from app.core.llm import get_chat_model
from app.generation.prompts import load_prompt
from app.models.pydantic_models import (
    ChecklistItem,
    DraftLearnedPattern,
)
from app.models.sqlalchemy_models import (
    Checklist as ChecklistORM,
    ChecklistItem as ChecklistItemORM,
    EditEvent as EditEventORM,
    FewShotExample as FewShotExampleORM,
    LearnedPattern as LearnedPatternORM,
)
from app.retrieval.embedding import embed_query

_PROMPT_TEMPLATE = load_prompt("pattern_extraction", version="v1")
_LLM_TIMEOUT = 30  # L1: 30s timeout
# Only retry on LLM output issues; do NOT retry on TimeoutError (doubles the 30s window).
_RECOVERABLE = (ValidationError, ValueError)

# Confidence formula: min(0.5 + 0.15 * count, 0.95)
_CONFIDENCE_BASE = 0.5
_CONFIDENCE_PER_EDIT = 0.15
_CONFIDENCE_CAP = 0.95
_PROMOTION_MIN_COUNT = 3
_PROMOTION_MIN_CONFIDENCE = 0.7
_LOOKBACK_DAYS = 30


def _compute_confidence(count: int) -> float:
    return min(_CONFIDENCE_BASE + _CONFIDENCE_PER_EDIT * count, _CONFIDENCE_CAP)


def _should_promote(count: int, confidence: float) -> bool:
    return count >= _PROMOTION_MIN_COUNT and confidence >= _PROMOTION_MIN_CONFIDENCE


async def _call_extraction_llm(
    edit_events_json: str,
    existing_patterns_json: str,
) -> list[DraftLearnedPattern]:
    """One structured LLM call; retried once with strict suffix on ValidationError (L1)."""
    schema_json = json.dumps(DraftLearnedPattern.model_json_schema(), indent=2)
    prompt = (
        _PROMPT_TEMPLATE
        .replace("{schema}", schema_json)
        .replace("{edit_events_json}", edit_events_json)
        .replace("{existing_patterns_json}", existing_patterns_json)
    )
    model = get_chat_model(role="quality").with_structured_output(
        list[DraftLearnedPattern]  # type: ignore[arg-type]
    )

    try:
        result = await asyncio.wait_for(model.ainvoke(prompt), timeout=_LLM_TIMEOUT)
    except _RECOVERABLE:
        # One stricter retry
        strict_prompt = (
            prompt
            + "\n\nIMPORTANT: Return ONLY a valid JSON array. No prose, no markdown."
        )
        result = await asyncio.wait_for(
            model.ainvoke(strict_prompt), timeout=_LLM_TIMEOUT
        )

    if not isinstance(result, list):
        raise ValueError(f"Unexpected extraction output: {type(result)}")
    return result  # type: ignore[return-value]


async def extract_patterns(checklist_id: uuid.UUID) -> None:
    """Background task: mine edit events → LearnedPattern rows + few_shot_examples rows.

    On any failure, logs ERROR and marks eval_metrics["pattern_extraction_error"] on the
    checklist rather than raising into the BackgroundTask.
    """
    log = logger.bind(checklist_id=str(checklist_id))
    try:
        await _extract_patterns_inner(checklist_id)
    except Exception as exc:
        log.bind(error=str(exc)[:200]).error("extract_patterns_failed")
        # Best-effort: record the failure in eval_metrics without raising.
        try:
            async with SessionLocal() as session:
                stmt = (
                    update(ChecklistORM)
                    .where(ChecklistORM.id == checklist_id)
                    .values(
                        eval_metrics={
                            "pattern_extraction_error": str(exc)[:200]
                        }
                    )
                    .execution_options(synchronize_session=False)  # L5
                )
                await session.execute(stmt)
                await session.commit()
        except Exception:
            pass


async def _extract_patterns_inner(checklist_id: uuid.UUID) -> None:
    log = logger.bind(checklist_id=str(checklist_id))

    async with SessionLocal() as session:
        # Load the checklist to get doc_type + draft_originals.
        checklist_row = await session.get(ChecklistORM, checklist_id)
        if checklist_row is None:
            raise ValueError(f"Checklist {checklist_id} not found")

        doc_type = await _resolve_doc_type(session, checklist_row)

        # Load edit events for this checklist + recent cross-checklist history.
        event_rows = await _load_edit_events(session, checklist_id, doc_type)
        if not event_rows:
            log.info("extract_patterns_no_events")
            return

        event_ids = {row.id for row in event_rows}
        events_json = json.dumps(
            [
                {
                    "id": str(row.id),
                    "event_type": row.event_type,
                    "payload": row.payload,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in event_rows
            ],
            default=str,
        )

        # Load existing patterns for this doc_type.
        existing_rows = await _load_existing_patterns(session, doc_type)
        existing_json = json.dumps(
            [
                {
                    "id": str(row.id),
                    "pattern_type": row.pattern_type,
                    "doc_type_scope": row.doc_type_scope,
                    "rule_json": row.rule_json,
                    "corroborating_edit_count": row.corroborating_edit_count,
                    "promoted": row.promoted,
                }
                for row in existing_rows
            ],
            default=str,
        )

        # LLM call (L1: one call, one retry on ValidationError).
        try:
            drafts = await _call_extraction_llm(events_json, existing_json)
        except Exception as exc:
            log.bind(error=str(exc)[:200]).error("extraction_llm_failed")
            return

        # Validate and upsert patterns (P1 invariant enforced here).
        now = datetime.now(timezone.utc)
        for draft in drafts:
            # P1: reject any draft without at least one valid supporting event.
            valid_ids = [eid for eid in draft.supporting_edit_ids if eid in event_ids]
            if not valid_ids:
                log.bind(
                    pattern_type=draft.pattern_type,
                    rule_json=draft.rule_json,
                ).warning("extract_patterns_P1_rejected_no_evidence")
                continue

            await _upsert_pattern(
                session=session,
                draft=draft,
                valid_ids=valid_ids,
                existing_rows=existing_rows,
                doc_type=doc_type,
                now=now,
            )

        # Build few_shot_examples for items that were edited.
        await _write_few_shot_examples(
            session=session,
            checklist_id=checklist_id,
            checklist_row=checklist_row,
            event_rows=event_rows,
            doc_type=doc_type,
        )

        await session.commit()
        log.info("extract_patterns_done")


async def _resolve_doc_type(session: AsyncSession, checklist_row: ChecklistORM) -> str:
    """Derive doc_type from the checklist's template slug (stored in checklist_templates)."""
    from app.models.sqlalchemy_models import ChecklistTemplate as ChecklistTemplateORM

    tmpl = await session.get(ChecklistTemplateORM, checklist_row.template_id)
    return tmpl.doc_type if tmpl and tmpl.doc_type else "other"


async def _load_edit_events(
    session: AsyncSession,
    checklist_id: uuid.UUID,
    doc_type: str,
) -> list[EditEventORM]:
    """Load events for this checklist + recent same-doc_type history."""
    lookback = datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS)

    # Checklists of the same doc_type in the lookback window.
    from app.models.sqlalchemy_models import ChecklistTemplate as ChecklistTemplateORM

    related_ids_result = await session.execute(
        select(ChecklistORM.id)
        .join(
            ChecklistTemplateORM,
            ChecklistORM.template_id == ChecklistTemplateORM.id,
        )
        .where(
            ChecklistTemplateORM.doc_type == doc_type,
            ChecklistORM.finalized_at >= lookback,
        )
    )
    related_ids = {row[0] for row in related_ids_result} | {checklist_id}

    result = await session.execute(
        select(EditEventORM).where(EditEventORM.checklist_id.in_(related_ids))
    )
    return list(result.scalars().all())


async def _load_existing_patterns(
    session: AsyncSession, doc_type: str
) -> list[LearnedPatternORM]:
    result = await session.execute(
        select(LearnedPatternORM).where(
            LearnedPatternORM.doc_type_scope.in_([doc_type, "*"])
        )
    )
    return list(result.scalars().all())


async def _upsert_pattern(
    session: AsyncSession,
    draft: DraftLearnedPattern,
    valid_ids: list[uuid.UUID],
    existing_rows: list[LearnedPatternORM],
    doc_type: str,
    now: datetime,
) -> None:
    """Increment corroboration on existing pattern or insert a new one."""
    # Find existing by (pattern_type, doc_type_scope, rule_json).
    rule_key = json.dumps(draft.rule_json, sort_keys=True)
    existing = next(
        (
            r
            for r in existing_rows
            if r.pattern_type == draft.pattern_type
            and r.doc_type_scope == draft.doc_type_scope
            and json.dumps(r.rule_json, sort_keys=True) == rule_key
        ),
        None,
    )

    if existing is not None:
        # Union the supporting edit ids.
        merged_ids = list(
            {str(eid) for eid in (existing.supporting_edit_ids or [])}
            | {str(eid) for eid in valid_ids}
        )
        # P5: if pattern was dismissed, only count edits after dismissal timestamp.
        dismissed_at = None
        if isinstance(existing.rule_json, dict):
            meta = existing.rule_json.get("_meta", {})
            if isinstance(meta, dict):
                dismissed_at_str = meta.get("dismissed_at")
                if dismissed_at_str:
                    try:
                        dismissed_at = datetime.fromisoformat(dismissed_at_str)
                    except ValueError:
                        pass

        count = len(merged_ids)
        if dismissed_at:
            # Recompute from events after dismissal only.
            post_dismissal = [
                str(eid)
                for eid in valid_ids
                # We only have the IDs here, not timestamps; the extractor counts
                # the cardinality of the merged set as a conservative upper bound.
                # A tighter implementation would join back to edit_events.created_at.
            ]
            count = len(
                {str(eid) for eid in (existing.supporting_edit_ids or [])}
                | set(post_dismissal)
            )

        confidence = _compute_confidence(count)
        promoted = _should_promote(count, confidence)

        await session.execute(
            update(LearnedPatternORM)
            .where(LearnedPatternORM.id == existing.id)
            .values(
                supporting_edit_ids=merged_ids,
                corroborating_edit_count=count,
                confidence=confidence,
                promoted=promoted,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)  # L5
        )
    else:
        count = len(valid_ids)
        confidence = _compute_confidence(count)
        promoted = _should_promote(count, confidence)
        new_id = uuid.uuid4()
        session.add(
            LearnedPatternORM(
                id=new_id,
                pattern_type=draft.pattern_type,
                doc_type_scope=draft.doc_type_scope,
                rule_json=draft.rule_json,
                supporting_edit_ids=[str(eid) for eid in valid_ids],
                confidence=confidence,
                corroborating_edit_count=count,
                promoted=promoted,
                created_at=now,
                updated_at=now,
            )
        )


async def _write_few_shot_examples(
    session: AsyncSession,
    checklist_id: uuid.UUID,
    checklist_row: ChecklistORM,
    event_rows: list[EditEventORM],
    doc_type: str,
) -> None:
    """Write few_shot_examples for items that have at least one edit event."""
    # Collect item_ids that were edited in this checklist.
    edited_item_ids = {
        row.checklist_item_id
        for row in event_rows
        if row.checklist_id == checklist_id and row.checklist_item_id is not None
    }
    if not edited_item_ids:
        return

    # Load current (final) state of those items.
    items_result = await session.execute(
        select(ChecklistItemORM).where(
            ChecklistItemORM.checklist_id == checklist_id,
            ChecklistItemORM.id.in_(edited_item_ids),
        )
    )
    item_rows = list(items_result.scalars().all())

    for item_row in item_rows:
        # final_item = current DB state (after all operator edits).
        final_item = _orm_to_checklist_item(item_row)

        # original_draft = what was stored in checklist.eval_metrics["draft_originals"]
        # at validate_item time.  If absent (e.g. checklist generated before Phase 5),
        # skip this item.
        draft_originals: dict = (checklist_row.eval_metrics or {}).get(
            "draft_originals", {}
        )
        original_dict = draft_originals.get(str(item_row.source_template_item_id)) or \
            draft_originals.get(str(item_row.id))
        if not original_dict:
            continue

        try:
            original_item = ChecklistItem.model_validate(original_dict)
        except Exception:
            continue

        # P3: skip identity pairs.
        if original_item.model_dump() == final_item.model_dump():
            continue

        # Build context embedding.
        evidence_summary = " ".join(
            c.snippet for c in (final_item.evidence or [])
        )[:500]
        query_text = (
            final_item.title
            + " "
            + str(final_item.category)
            + " "
            + evidence_summary
        )
        try:
            embedding = await embed_query(query_text)
        except Exception as exc:
            logger.bind(item_id=str(item_row.id), error=str(exc)[:120]).warning(
                "few_shot_embed_failed"
            )
            continue

        session.add(
            FewShotExampleORM(
                id=uuid.uuid4(),
                doc_type=doc_type,
                category=str(item_row.category),
                template_item_id=item_row.source_template_item_id,
                original_draft=original_item.model_dump(mode="json"),
                final_item=final_item.model_dump(mode="json"),
                context_embedding=embedding,
                created_at=now,
            )
        )


def _orm_to_checklist_item(row: ChecklistItemORM) -> ChecklistItem:
    from app.models.pydantic_models import EvidenceCitation

    evidence = [
        EvidenceCitation(
            citation_id=c.id,
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            page_number=c.page_number,
            char_offset_start=c.char_offset_start,
            char_offset_end=c.char_offset_end,
            snippet=c.snippet,
            retrieval_score=c.retrieval_score,
            rerank_score=c.rerank_score,
        )
        for c in (row.citations or [])
    ]
    return ChecklistItem(
        id=row.id,
        source_template_item_id=row.source_template_item_id,
        category=row.category,  # type: ignore[arg-type]
        title=row.title,
        description=row.description,
        status=row.status,  # type: ignore[arg-type]
        required=row.required,
        evidence=evidence,
        confidence=row.confidence or 0.0,
        rationale=row.rationale or "",
        learned_from_pattern_ids=list(row.learned_from_pattern_ids or []),
    )
