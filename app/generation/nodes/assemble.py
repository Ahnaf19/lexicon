"""assemble node — persists the completed Checklist to Postgres and returns it."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select

from app.core.db import SessionLocal
from app.generation.state import ChecklistState
from app.models.pydantic_models import Checklist, ChecklistItem, EvidenceCitation
from app.models.sqlalchemy_models import Checklist as ChecklistORM
from app.models.sqlalchemy_models import ChecklistItem as ChecklistItemORM
from app.models.sqlalchemy_models import EvidenceCitation as EvidenceCitationORM


async def assemble(state: ChecklistState) -> dict[str, object]:
    """Build and persist the final Checklist; return checklist_id and pydantic model."""
    template = state["template"]
    case_id = state["case_id"]
    in_progress = state.get("items_in_progress") or {}
    model_version = state.get("model_version", "unknown")
    prompt_version = state.get("prompt_version", "v1")

    # Collect final ChecklistItems in template order.
    items: list[ChecklistItem] = []
    for item_def in template.items:
        entry = in_progress.get(item_def.slug)
        if isinstance(entry, ChecklistItem):
            items.append(entry)
        else:
            # Fallback: shouldn't happen, but surface as unclear rather than dropping.
            items.append(
                ChecklistItem(
                    id=uuid.uuid4(),
                    source_template_item_id=item_def.stable_uuid(template.id),
                    category=item_def.category,
                    title=item_def.title,
                    description=item_def.description,
                    status="unclear",
                    required=item_def.required,
                    evidence=[],
                    confidence=0.0,
                    rationale="Item not processed.",
                    learned_from_pattern_ids=[],
                )
            )

    checklist_id = uuid.uuid4()
    generated_at = datetime.now(timezone.utc)

    async with SessionLocal() as session:
        # Insert Checklist row.
        checklist_orm = ChecklistORM(
            id=checklist_id,
            case_id=case_id,
            template_id=template.id,
            status="draft",
            generated_at=generated_at,
            model_version=model_version,
            prompt_version=prompt_version,
        )
        session.add(checklist_orm)
        await session.flush()  # get the FK before inserting items

        for item in items:
            item_orm = ChecklistItemORM(
                id=item.id,
                checklist_id=checklist_id,
                source_template_item_id=item.source_template_item_id,
                category=item.category,
                title=item.title,
                description=item.description,
                status=item.status,
                required=item.required,
                confidence=item.confidence,
                rationale=item.rationale,
                learned_from_pattern_ids=[],
            )
            session.add(item_orm)
            await session.flush()

            for ec in item.evidence:
                session.add(
                    EvidenceCitationORM(
                        id=ec.citation_id,
                        checklist_item_id=item.id,
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

        await session.commit()

    logger.bind(
        checklist_id=str(checklist_id),
        case_id=str(case_id),
        template_slug=template.slug,
        item_count=len(items),
        present_count=sum(1 for i in items if i.status == "present"),
        unclear_count=sum(1 for i in items if i.status == "unclear"),
        missing_count=sum(1 for i in items if i.status == "missing"),
    ).info("assemble_checklist_persisted")

    checklist = Checklist(
        id=checklist_id,
        case_id=case_id,
        template_id=template.id,
        items=items,
        generated_at=generated_at,
        model_version=model_version,
        prompt_version=prompt_version,
    )

    return {"checklist": checklist, "checklist_id": checklist_id}
