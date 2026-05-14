"""load_template node — loads the in-code template and upserts its DB row."""

from __future__ import annotations

from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from app.core.db import SessionLocal
from app.generation.state import ChecklistState
from app.generation.templates import TEMPLATES
from app.models.sqlalchemy_models import ChecklistTemplate as ChecklistTemplateORM


async def load_template(state: ChecklistState) -> dict[str, object]:
    """Load template from registry; upsert a checklist_templates row (idempotent)."""
    slug = state["template_slug"]
    template = TEMPLATES.get(slug)
    if template is None:
        raise ValueError(f"Unknown template slug: {slug!r}. Available: {list(TEMPLATES)}")

    # Upsert into checklist_templates so Checklist.template_id FK resolves.
    async with SessionLocal() as session:
        stmt = (
            insert(ChecklistTemplateORM)
            .values(
                id=template.id,
                name=template.name,
                doc_type=template.doc_type,
                version=template.version,
                items=[item.model_dump() for item in template.items],
                active=True,
            )
            .on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "name": template.name,
                    "version": template.version,
                    "items": [item.model_dump() for item in template.items],
                    "active": True,
                },
            )
        )
        await session.execute(stmt)
        await session.commit()

    logger.bind(template_slug=slug, item_count=len(template.items)).info("load_template_ok")

    return {
        "template": template,
        "learned_patterns": [],
        "items_in_progress": {item.slug: None for item in template.items},
        "search_hits_by_item": {},
        "errors": [],
        "prompt_version": "v1",
        "item_index": 0,
        "current_item_slug": template.items[0].slug if template.items else None,
    }
