"""load_template node — loads the in-code template, upserts its DB row, applies promoted patterns."""

from __future__ import annotations

import copy

from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.db import SessionLocal
from app.generation.state import ChecklistState
from app.generation.templates import TEMPLATES
from app.generation.templates.types import TemplateItem
from app.models.pydantic_models import (
    CategoryRemap as CategoryRemapRule,
    LearnedPattern,
    RenameRule,
    StylePreference as StylePreferenceRule,
    TemplateAddition,
    TemplateRemoval,
)
from app.models.sqlalchemy_models import ChecklistTemplate as ChecklistTemplateORM
from app.models.sqlalchemy_models import LearnedPattern as LearnedPatternORM


async def load_template(state: ChecklistState) -> dict[str, object]:
    """Load template from registry; upsert a checklist_templates row; apply promoted patterns."""
    slug = state["template_slug"]
    template = TEMPLATES.get(slug)
    if template is None:
        raise ValueError(f"Unknown template slug: {slug!r}. Available: {list(TEMPLATES)}")

    async with SessionLocal() as session:
        # Upsert into checklist_templates so Checklist.template_id FK resolves.
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

        # Load promoted patterns scoped to this doc_type or wildcard.
        patterns_result = await session.execute(
            select(LearnedPatternORM).where(
                LearnedPatternORM.doc_type_scope.in_([template.doc_type, "*"]),
                LearnedPatternORM.promoted.is_(True),
            )
        )
        pattern_rows = list(patterns_result.scalars().all())

    learned_patterns = [
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
        for row in pattern_rows
    ]

    # Mutate a copy of the template to apply template_addition / template_removal patterns.
    mutated_items = copy.copy(list(template.items))
    added_count = 0
    removed_count = 0

    for pattern in learned_patterns:
        if pattern.pattern_type == "template_addition":
            try:
                rule = TemplateAddition.model_validate(pattern.rule_json)
                mutated_items.append(rule.item_template)
                added_count += 1
            except Exception as exc:
                logger.bind(pattern_id=str(pattern.id), error=str(exc)[:80]).warning(
                    "load_template_addition_invalid"
                )
        elif pattern.pattern_type == "template_removal":
            try:
                rule = TemplateRemoval.model_validate(pattern.rule_json)
                mutated_items = [i for i in mutated_items if i.slug != rule.item_slug]
                removed_count += 1
            except Exception as exc:
                logger.bind(pattern_id=str(pattern.id), error=str(exc)[:80]).warning(
                    "load_template_removal_invalid"
                )

    # Reconstruct template with mutated items so downstream nodes see the evolved template.
    from app.generation.templates.types import ChecklistTemplate

    active_template = ChecklistTemplate(
        slug=template.slug,
        id=template.id,
        name=template.name,
        doc_type=template.doc_type,
        version=template.version,
        items=mutated_items,
    )

    logger.bind(
        template_slug=slug,
        original_items=len(template.items),
        added=added_count,
        removed=removed_count,
        promoted_patterns=len(learned_patterns),
    ).info("load_template_ok")

    return {
        "template": active_template,
        "learned_patterns": learned_patterns,
        "items_in_progress": {item.slug: None for item in active_template.items},
        "search_hits_by_item": {},
        "errors": [],
        "prompt_version": "v1",
        "item_index": 0,
        "current_item_slug": active_template.items[0].slug if active_template.items else None,
        "draft_originals": {},
    }
