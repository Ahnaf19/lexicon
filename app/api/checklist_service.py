"""Checklist item service layer — pure async helpers shared by API routes and the eval.

No FastAPI imports. All helpers flush but do NOT commit; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.learning.edit_capture import write_edit_events_bulk
from app.models.pydantic_models import (
    CategoryReordered,
    ChecklistItem,
    ItemAdded,
    ItemRemoved,
    ItemRenamed,
    ItemTextRewritten,
    PartialChecklistItem,
    RequiredToggled,
    StatusChanged,
)
from app.models.sqlalchemy_models import ChecklistItem as ChecklistItemORM
from app.models.sqlalchemy_models import EditEvent as EditEventORM
from app.models.sqlalchemy_models import EvidenceCitation as EvidenceCitationORM


async def apply_item_patch(
    session: AsyncSession,
    checklist_id: uuid.UUID,
    item_id: uuid.UUID,
    partial: PartialChecklistItem,
    actor: str | None,
) -> None:
    """Diff partial against current state, mutate ORM fields, bulk-insert edit events.

    Raises ValueError if item not found. Does not commit.
    """
    result = await session.execute(
        select(ChecklistItemORM).where(
            ChecklistItemORM.id == item_id,
            ChecklistItemORM.checklist_id == checklist_id,
        )
    )
    item_orm = result.scalar_one_or_none()
    if item_orm is None:
        raise ValueError(f"Item {item_id} not found in checklist {checklist_id}")

    events = []

    if partial.title is not None and partial.title != item_orm.title:
        events.append(
            ItemRenamed(item_id=item_id, old_title=item_orm.title, new_title=partial.title)
        )
        item_orm.title = partial.title

    if partial.status is not None and partial.status != item_orm.status:
        events.append(
            StatusChanged(
                item_id=item_id,
                old_status=item_orm.status,  # type: ignore[arg-type]
                new_status=partial.status,
            )
        )
        item_orm.status = partial.status

    if partial.category is not None and partial.category != item_orm.category:
        events.append(
            CategoryReordered(
                item_id=item_id,
                old_category=item_orm.category,
                new_category=partial.category,
            )
        )
        item_orm.category = partial.category

    if partial.description is not None and partial.description != item_orm.description:
        events.append(
            ItemTextRewritten(
                item_id=item_id,
                field="description",
                old_text=item_orm.description,
                new_text=partial.description,
            )
        )
        item_orm.description = partial.description

    if partial.rationale is not None and partial.rationale != (item_orm.rationale or ""):
        events.append(
            ItemTextRewritten(
                item_id=item_id,
                field="rationale",
                old_text=item_orm.rationale or "",
                new_text=partial.rationale,
            )
        )
        item_orm.rationale = partial.rationale

    if partial.required is not None and partial.required != item_orm.required:
        events.append(
            RequiredToggled(item_id=item_id, old=item_orm.required, new=partial.required)
        )
        item_orm.required = partial.required

    if partial.confidence is not None:
        item_orm.confidence = partial.confidence

    if events:
        await write_edit_events_bulk(
            session=session,
            checklist_id=checklist_id,
            item_id=item_id,
            events=events,
            actor=actor,
        )


async def add_item(
    session: AsyncSession,
    checklist_id: uuid.UUID,
    item: ChecklistItem,
    actor: str | None,
) -> uuid.UUID:
    """Insert a new ChecklistItem row plus an item_added event. Returns the new item UUID.

    Does not commit.
    """
    now = datetime.now(timezone.utc)
    new_id = uuid.uuid4()

    item_orm = ChecklistItemORM(
        id=new_id,
        checklist_id=checklist_id,
        source_template_item_id=item.source_template_item_id,
        category=item.category,
        title=item.title,
        description=item.description,
        status=item.status,
        required=item.required,
        confidence=item.confidence,
        rationale=item.rationale,
        learned_from_pattern_ids=list(item.learned_from_pattern_ids),
    )
    session.add(item_orm)
    await session.flush()  # get PK before evidence citations

    for ec in item.evidence:
        session.add(
            EvidenceCitationORM(
                id=ec.citation_id,
                checklist_item_id=new_id,
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

    final_item = item.model_copy(update={"id": new_id})
    event = ItemAdded(item=final_item)
    session.add(
        EditEventORM(
            id=uuid.uuid4(),
            checklist_id=checklist_id,
            checklist_item_id=new_id,
            event_type=event.event_type,
            payload=event.model_dump(mode="json"),
            actor=actor,
            created_at=now,
        )
    )
    return new_id


async def delete_item(
    session: AsyncSession,
    checklist_id: uuid.UUID,
    item_id: uuid.UUID,
    actor: str | None,
) -> None:
    """Write item_removed event, null out FK references, hard-delete the item row.

    Raises ValueError if item not found. Does not commit.
    """
    result = await session.execute(
        select(ChecklistItemORM).where(
            ChecklistItemORM.id == item_id,
            ChecklistItemORM.checklist_id == checklist_id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise ValueError(f"Item {item_id} not found in checklist {checklist_id}")

    now = datetime.now(timezone.utc)
    event = ItemRemoved(item_id=item_id)
    session.add(
        EditEventORM(
            id=uuid.uuid4(),
            checklist_id=checklist_id,
            checklist_item_id=None,
            event_type=event.event_type,
            payload=event.model_dump(mode="json"),
            actor=actor,
            created_at=now,
        )
    )
    await session.execute(
        update(EditEventORM)
        .where(EditEventORM.checklist_item_id == item_id)
        .values(checklist_item_id=None)
        .execution_options(synchronize_session=False)
    )
    await session.execute(
        delete(ChecklistItemORM)
        .where(ChecklistItemORM.id == item_id)
        .execution_options(synchronize_session=False)
    )
