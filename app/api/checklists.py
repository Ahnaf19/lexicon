"""Checklist generation and retrieval endpoints (PRD §5e)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.db import SessionLocal
from app.generation.graph import get_compiled_graph
from app.generation.state import ChecklistState
from app.models.pydantic_models import Checklist, ChecklistItem, EvidenceCitation
from app.models.sqlalchemy_models import Checklist as ChecklistORM
from app.models.sqlalchemy_models import ChecklistItem as ChecklistItemORM
from app.models.sqlalchemy_models import EvidenceCitation as EvidenceCitationORM

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
