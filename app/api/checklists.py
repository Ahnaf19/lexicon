"""Checklist generation, retrieval, and mutation endpoints (PRD §5e + §5f)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.checklist_service import (
    add_item as _svc_add,
    apply_item_patch as _svc_patch,
    delete_item as _svc_delete,
)
from app.core.db import SessionLocal, get_session
from app.generation.graph import get_compiled_graph
from app.generation.state import ChecklistState
from app.learning.edit_capture import get_operator_id
from app.learning.pattern_extractor import extract_patterns
from app.models.pydantic_models import (
    Checklist,
    ChecklistItem,
    ChecklistSummary,
    EvidenceCitation,
    EvidenceMutation,
    LearnedPattern,
    PartialChecklistItem,
    EvidenceAdded,
    EvidenceCorrected,
)
from app.models.sqlalchemy_models import Checklist as ChecklistORM
from app.models.sqlalchemy_models import ChecklistItem as ChecklistItemORM
from app.models.sqlalchemy_models import EditEvent as EditEventORM
from app.models.sqlalchemy_models import EvidenceCitation as EvidenceCitationORM
from app.models.sqlalchemy_models import LearnedPattern as LearnedPatternORM

router = APIRouter()


class GenerateRequest(BaseModel):
    case_id: uuid.UUID
    template_slug: str | None = None


# ---------------------------------------------------------------------------
# SSE event helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:  # type: ignore[type-arg]
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_generation(case_id: uuid.UUID, template_slug: str | None) -> AsyncGenerator[str, None]:
    initial_state: ChecklistState = {
        "case_id": case_id,
        "template_slug": template_slug or "",
        "document_ids": [],
        "errors": [],
    }
    # Inject template_slug only if provided; classify_doc_set reads the empty string as "no override".
    if not template_slug:
        initial_state.pop("template_slug", None)  # type: ignore[misc]

    graph = get_compiled_graph()

    _PROGRESS_NODES = frozenset(
        {"retrieve_evidence", "draft_item", "validate_item", "critique", "assemble"}
    )
    checklist_id: str | None = None

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            if kind == "on_chain_start" and name in _PROGRESS_NODES:
                payload: dict = {"node": name, "timestamp": datetime.now(timezone.utc).isoformat()}
                inp = data.get("input") or {}
                if isinstance(inp, dict):
                    payload["item_slug"] = inp.get("current_item_slug")
                yield _sse("node_start", payload)

            elif kind == "on_chain_end" and name in _PROGRESS_NODES:
                payload = {"node": name, "timestamp": datetime.now(timezone.utc).isoformat()}
                out = data.get("output") or {}
                if isinstance(out, dict):
                    payload["item_slug"] = out.get("current_item_slug")
                    if name == "assemble" and "checklist_id" in out:
                        checklist_id = str(out["checklist_id"])
                        payload["checklist_id"] = checklist_id
                yield _sse("node_end", payload)

        yield _sse("done", {"status": "complete", "checklist_id": checklist_id})

    except ValueError as exc:
        yield _sse("error", {"detail": str(exc)})
    except Exception as exc:
        yield _sse("error", {"detail": f"Internal error: {exc}"})


# ---------------------------------------------------------------------------
# POST /checklists/generate — SSE stream
# ---------------------------------------------------------------------------


@router.post("/generate")
async def generate_checklist(req: GenerateRequest) -> StreamingResponse:
    """Start checklist generation; streams node-level progress via SSE."""
    return StreamingResponse(
        _stream_generation(req.case_id, req.template_slug),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# GET /learned-patterns — must be before /{checklist_id} (static before parameterized)
# ---------------------------------------------------------------------------


@router.get("/learned-patterns", response_model=list[LearnedPattern])
async def list_learned_patterns_route(
    doc_type: str | None = None,
    promoted: bool | None = None,
    pattern_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
) -> list[LearnedPattern]:
    """List learned patterns with optional filters."""
    from sqlalchemy import select as _select

    stmt = _select(LearnedPatternORM)
    if doc_type is not None:
        stmt = stmt.where(LearnedPatternORM.doc_type_scope == doc_type)
    if promoted is not None:
        stmt = stmt.where(LearnedPatternORM.promoted.is_(promoted))
    if pattern_type is not None:
        stmt = stmt.where(LearnedPatternORM.pattern_type == pattern_type)
    stmt = stmt.order_by(LearnedPatternORM.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    return [
        LearnedPattern(
            id=row.id,
            pattern_type=row.pattern_type,  # type: ignore[arg-type]
            doc_type_scope=row.doc_type_scope,
            rule_json=row.rule_json or {},
            supporting_edit_ids=list(row.supporting_edit_ids or []),
            confidence=row.confidence or 0.0,
            corroborating_edit_count=row.corroborating_edit_count or 0,
            promoted=row.promoted or False,
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.post("/learned-patterns/{pattern_id}/dismiss", response_model=LearnedPattern)
async def dismiss_learned_pattern_route(
    pattern_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> LearnedPattern:
    """Dismiss a pattern: decrement corroboration, demote if below threshold.

    P5: dismissal timestamp stored in rule_json["_meta"]["dismissed_at"] so that
    re-promotion requires NEW edits after this timestamp.
    """
    from sqlalchemy import update as sa_update

    row = await session.get(LearnedPatternORM, pattern_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

    new_count = max(0, (row.corroborating_edit_count or 1) - 1)
    from app.learning.pattern_extractor import _compute_confidence, _should_promote

    new_confidence = _compute_confidence(new_count)
    new_promoted = _should_promote(new_count, new_confidence)

    rule_json = dict(row.rule_json or {})
    meta = dict(rule_json.get("_meta", {}))
    meta["dismissed_at"] = datetime.now(timezone.utc).isoformat()
    rule_json["_meta"] = meta

    await session.execute(
        sa_update(LearnedPatternORM)
        .where(LearnedPatternORM.id == pattern_id)
        .values(
            corroborating_edit_count=new_count,
            confidence=new_confidence,
            promoted=new_promoted,
            rule_json=rule_json,
            updated_at=datetime.now(timezone.utc),
        )
        .execution_options(synchronize_session=False)  # L5
    )
    await session.commit()
    await session.refresh(row)

    return LearnedPattern(
        id=row.id,
        pattern_type=row.pattern_type,  # type: ignore[arg-type]
        doc_type_scope=row.doc_type_scope,
        rule_json=row.rule_json or {},
        supporting_edit_ids=list(row.supporting_edit_ids or []),
        confidence=row.confidence or 0.0,
        corroborating_edit_count=row.corroborating_edit_count or 0,
        promoted=row.promoted or False,
        created_at=row.created_at,
    )


# ---------------------------------------------------------------------------
# GET /cases/{case_id}/checklists — list checklists for a case (static before parameterized)
# ---------------------------------------------------------------------------


@router.get("/cases/{case_id}/checklists", response_model=list[ChecklistSummary])
async def list_case_checklists(
    case_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> list[ChecklistSummary]:
    """List checklists generated for a case, newest first. UI helper (PRD §9)."""
    stmt = (
        select(ChecklistORM)
        .where(ChecklistORM.case_id == case_id)
        .options(selectinload(ChecklistORM.items))
        .order_by(ChecklistORM.generated_at.desc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [
        ChecklistSummary(
            checklist_id=row.id,
            case_id=row.case_id,
            generated_at=row.generated_at,
            model_version=row.model_version or "unknown",
            prompt_version=row.prompt_version or "v1",
            item_count=len(row.items or []),
            status=row.status,
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# GET /checklists/{id} — fetch persisted checklist
# ---------------------------------------------------------------------------


@router.get("/{checklist_id}", response_model=Checklist)
async def get_checklist(checklist_id: uuid.UUID) -> Checklist:
    """Fetch a persisted checklist with all items and evidence."""
    async with SessionLocal() as session:
        result = await session.execute(
            select(ChecklistORM)
            .where(ChecklistORM.id == checklist_id)
            .options(
                selectinload(ChecklistORM.items).selectinload(ChecklistItemORM.citations)
            )
        )
        row = result.scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Checklist {checklist_id} not found")

    items = []
    for item_orm in (row.items or []):
        evidence = [
            EvidenceCitation(
                citation_id=ec.id,
                chunk_id=ec.chunk_id,
                doc_id=ec.doc_id,
                page_number=ec.page_number,
                char_offset_start=ec.char_offset_start,
                char_offset_end=ec.char_offset_end,
                snippet=ec.snippet,
                retrieval_score=ec.retrieval_score,
                rerank_score=ec.rerank_score,
            )
            for ec in (item_orm.citations or [])
        ]
        items.append(
            ChecklistItem(
                id=item_orm.id,
                source_template_item_id=item_orm.source_template_item_id,
                category=item_orm.category,  # type: ignore[arg-type]
                title=item_orm.title,
                description=item_orm.description,
                status=item_orm.status,  # type: ignore[arg-type]
                required=item_orm.required,
                evidence=evidence,
                confidence=item_orm.confidence or 0.0,
                rationale=item_orm.rationale or "",
                learned_from_pattern_ids=[],
            )
        )

    return Checklist(
        id=row.id,
        case_id=row.case_id,
        template_id=row.template_id,
        items=items,
        generated_at=row.generated_at or datetime.now(timezone.utc),
        model_version=row.model_version or "unknown",
        prompt_version=row.prompt_version or "v1",
        eval_metrics=row.eval_metrics,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item_orm_to_pydantic(item_orm: ChecklistItemORM) -> ChecklistItem:
    evidence = [
        EvidenceCitation(
            citation_id=ec.id,
            chunk_id=ec.chunk_id,
            doc_id=ec.doc_id,
            page_number=ec.page_number,
            char_offset_start=ec.char_offset_start,
            char_offset_end=ec.char_offset_end,
            snippet=ec.snippet,
            retrieval_score=ec.retrieval_score,
            rerank_score=ec.rerank_score,
        )
        for ec in (item_orm.citations or [])
    ]
    return ChecklistItem(
        id=item_orm.id,
        source_template_item_id=item_orm.source_template_item_id,
        category=item_orm.category,  # type: ignore[arg-type]
        title=item_orm.title,
        description=item_orm.description,
        status=item_orm.status,  # type: ignore[arg-type]
        required=item_orm.required,
        evidence=evidence,
        confidence=item_orm.confidence or 0.0,
        rationale=item_orm.rationale or "",
        learned_from_pattern_ids=list(item_orm.learned_from_pattern_ids or []),
    )


async def _require_checklist(session: AsyncSession, checklist_id: uuid.UUID) -> ChecklistORM:
    row = await session.get(ChecklistORM, checklist_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Checklist {checklist_id} not found")
    return row


async def _require_item(
    session: AsyncSession, checklist_id: uuid.UUID, item_id: uuid.UUID
) -> ChecklistItemORM:
    result = await session.execute(
        select(ChecklistItemORM)
        .where(
            ChecklistItemORM.id == item_id,
            ChecklistItemORM.checklist_id == checklist_id,
        )
        .options(selectinload(ChecklistItemORM.citations))
    )
    item_orm = result.scalar_one_or_none()
    if item_orm is None:
        raise HTTPException(
            status_code=404,
            detail=f"Item {item_id} not found in checklist {checklist_id}",
        )
    return item_orm


# ---------------------------------------------------------------------------
# PATCH /checklists/{id}/items/{item_id} — per-field mutation + edit events
# ---------------------------------------------------------------------------


@router.patch("/{checklist_id}/items/{item_id}", response_model=ChecklistItem)
async def patch_checklist_item(
    checklist_id: uuid.UUID,
    item_id: uuid.UUID,
    body: PartialChecklistItem,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ChecklistItem:
    """Apply partial updates to one item; emit one edit_event per changed field."""
    await _require_checklist(session, checklist_id)
    actor = get_operator_id(request)
    try:
        await _svc_patch(session, checklist_id, item_id, body, actor)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    await session.commit()
    item_orm = await _require_item(session, checklist_id, item_id)
    return _item_orm_to_pydantic(item_orm)


# ---------------------------------------------------------------------------
# POST /checklists/{id}/items — add a new item
# ---------------------------------------------------------------------------


@router.post("/{checklist_id}/items", response_model=ChecklistItem, status_code=201)
async def add_checklist_item(
    checklist_id: uuid.UUID,
    body: ChecklistItem,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ChecklistItem:
    """Insert a new item into the checklist and record an item_added event."""
    await _require_checklist(session, checklist_id)
    actor = get_operator_id(request)
    new_id = await _svc_add(session, checklist_id, body, actor)
    await session.commit()
    item_orm = await _require_item(session, checklist_id, new_id)
    return _item_orm_to_pydantic(item_orm)


# ---------------------------------------------------------------------------
# DELETE /checklists/{id}/items/{item_id} — hard delete + item_removed event
# ---------------------------------------------------------------------------


@router.delete("/{checklist_id}/items/{item_id}", status_code=204)
async def delete_checklist_item(
    checklist_id: uuid.UUID,
    item_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Hard-delete an item and record an item_removed event (capture before delete)."""
    await _require_checklist(session, checklist_id)
    actor = get_operator_id(request)
    try:
        await _svc_delete(session, checklist_id, item_id, actor)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    await session.commit()
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# PATCH /checklists/{id}/items/{item_id}/evidence — add/correct/remove evidence
# ---------------------------------------------------------------------------


@router.patch("/{checklist_id}/items/{item_id}/evidence", response_model=ChecklistItem)
async def mutate_evidence(
    checklist_id: uuid.UUID,
    item_id: uuid.UUID,
    body: EvidenceMutation,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ChecklistItem:
    """Add, correct, or remove an evidence citation and record the corresponding event."""
    await _require_checklist(session, checklist_id)
    item_orm = await _require_item(session, checklist_id, item_id)
    actor = get_operator_id(request)
    now = datetime.now(timezone.utc)

    if body.action == "add":
        if body.evidence is None:
            raise HTTPException(status_code=422, detail="evidence required for action=add")
        ec = body.evidence
        session.add(
            EvidenceCitationORM(
                id=ec.citation_id,
                checklist_item_id=item_id,
                chunk_id=ec.chunk_id,
                doc_id=ec.doc_id,
                page_number=ec.page_number,
                char_offset_start=ec.char_offset_start,
                char_offset_end=ec.char_offset_end,
                snippet=ec.snippet,
                retrieval_score=ec.retrieval_score,
                rerank_score=ec.rerank_score,
            )
        )
        event = EvidenceAdded(item_id=item_id, evidence=ec)

    elif body.action == "correct":
        if body.evidence is None or body.old_citation_id is None:
            raise HTTPException(
                status_code=422,
                detail="evidence and old_citation_id required for action=correct",
            )
        await session.execute(
            delete(EvidenceCitationORM).where(
                EvidenceCitationORM.id == body.old_citation_id,
                EvidenceCitationORM.checklist_item_id == item_id,
            )
        )
        ec = body.evidence
        session.add(
            EvidenceCitationORM(
                id=ec.citation_id,
                checklist_item_id=item_id,
                chunk_id=ec.chunk_id,
                doc_id=ec.doc_id,
                page_number=ec.page_number,
                char_offset_start=ec.char_offset_start,
                char_offset_end=ec.char_offset_end,
                snippet=ec.snippet,
                retrieval_score=ec.retrieval_score,
                rerank_score=ec.rerank_score,
            )
        )
        event = EvidenceCorrected(
            item_id=item_id,
            old_evidence=EvidenceCitation(
                citation_id=body.old_citation_id,
                chunk_id=uuid.UUID(int=0),
                doc_id=uuid.UUID(int=0),
                page_number=0,
                char_offset_start=0,
                char_offset_end=0,
                snippet="",
                retrieval_score=0.0,
            ),
            new_evidence=ec,
        )

    elif body.action == "remove":
        if body.old_citation_id is None:
            raise HTTPException(
                status_code=422, detail="old_citation_id required for action=remove"
            )
        await session.execute(
            delete(EvidenceCitationORM).where(
                EvidenceCitationORM.id == body.old_citation_id,
                EvidenceCitationORM.checklist_item_id == item_id,
            )
        )
        event = EvidenceAdded(
            item_id=item_id,
            evidence=EvidenceCitation(
                citation_id=body.old_citation_id,
                chunk_id=uuid.UUID(int=0),
                doc_id=uuid.UUID(int=0),
                page_number=0,
                char_offset_start=0,
                char_offset_end=0,
                snippet="",
                retrieval_score=0.0,
            ),
        )
    else:
        raise HTTPException(status_code=422, detail=f"Unknown action: {body.action}")

    session.add(
        EditEventORM(
            id=uuid.uuid4(),
            checklist_id=checklist_id,
            checklist_item_id=item_id,
            event_type=event.event_type,
            payload=event.model_dump(mode="json"),
            actor=actor,
            created_at=now,
        )
    )
    await session.commit()
    item_orm = await _require_item(session, checklist_id, item_id)
    return _item_orm_to_pydantic(item_orm)


# ---------------------------------------------------------------------------
# POST /checklists/{id}/finalize — set finalized_at + trigger pattern extraction
# ---------------------------------------------------------------------------


@router.post("/{checklist_id}/finalize", status_code=202)
async def finalize_checklist(
    checklist_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """Mark the checklist as finalized and start background pattern extraction."""
    checklist_row = await _require_checklist(session, checklist_id)
    checklist_row.finalized_at = datetime.now(timezone.utc)
    checklist_row.status = "finalized"
    await session.commit()

    background_tasks.add_task(extract_patterns, checklist_id)
    return {"checklist_id": str(checklist_id), "status": "extraction_pending"}

