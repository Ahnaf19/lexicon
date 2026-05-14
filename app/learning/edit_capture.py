"""Edit capture — append-only write of edit_events rows (PRD §5f Layer 1)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import Request
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pydantic_models import EditEvent
from app.models.sqlalchemy_models import EditEvent as EditEventORM

# Fallback actor when no X-Operator-Id header is present.
# Phase 6 wires real auth; for now any header-free call gets this sentinel.
_ANONYMOUS_OPERATOR_UUID = "anonymous-00000000-0000-0000-0000-000000000000"


def get_operator_id(request: Request) -> str:
    """FastAPI dependency: read X-Operator-Id header, fall back to anonymous sentinel."""
    return request.headers.get("X-Operator-Id", _ANONYMOUS_OPERATOR_UUID)


async def write_edit_event(
    session: AsyncSession,
    checklist_id: uuid.UUID,
    item_id: uuid.UUID | None,
    event: EditEvent,
    actor: str | None,
) -> uuid.UUID:
    """Append one edit_event row within the caller's transaction.

    Never UPDATE or DELETE — edit_events are immutable once written.
    """
    row_id = uuid.uuid4()
    payload = event.model_dump(mode="json")
    row = EditEventORM(
        id=row_id,
        checklist_id=checklist_id,
        checklist_item_id=item_id,
        event_type=event.event_type,
        payload=payload,
        actor=actor,
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    logger.bind(
        checklist_id=str(checklist_id),
        item_id=str(item_id) if item_id else None,
        event_type=event.event_type,
    ).debug("edit_event_appended")
    return row_id


async def write_edit_events_bulk(
    session: AsyncSession,
    checklist_id: uuid.UUID,
    item_id: uuid.UUID | None,
    events: list[EditEvent],
    actor: str | None,
) -> list[uuid.UUID]:
    """Bulk-append multiple edit events in one statement (L6).

    All rows share the same checklist_id / item_id / actor.
    """
    if not events:
        return []
    now = datetime.now(timezone.utc)
    rows = []
    ids: list[uuid.UUID] = []
    for event in events:
        row_id = uuid.uuid4()
        ids.append(row_id)
        rows.append(
            {
                "id": row_id,
                "checklist_id": checklist_id,
                "checklist_item_id": item_id,
                "event_type": event.event_type,
                "payload": event.model_dump(mode="json"),
                "actor": actor,
                "created_at": now,
            }
        )
    await session.execute(
        EditEventORM.__table__.insert(),  # type: ignore[union-attr]
        rows,
    )
    logger.bind(
        checklist_id=str(checklist_id),
        count=len(rows),
    ).debug("edit_events_bulk_appended")
    return ids
