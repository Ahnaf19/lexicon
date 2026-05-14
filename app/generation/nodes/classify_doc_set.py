"""classify_doc_set node — picks the template based on indexed document doc_types."""

from __future__ import annotations

import uuid
from collections import Counter

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import SessionLocal
from app.generation.state import ChecklistState
from app.models.sqlalchemy_models import Document


async def classify_doc_set(state: ChecklistState) -> dict[str, object]:
    """Read doc_type from indexed documents; majority-vote to pick a template slug."""
    case_id: uuid.UUID = state["case_id"]
    explicit_slug: str | None = state.get("template_slug")

    async with SessionLocal() as session:
        result = await session.execute(
            select(Document.id, Document.doc_type).where(
                Document.case_id == case_id,
                Document.status == "indexed",
            )
        )
        rows = result.all()

    if not rows:
        raise ValueError(f"No indexed documents found for case_id={case_id}")

    doc_ids = [row[0] for row in rows]
    doc_types = [row[1] for row in rows if row[1]  ]

    if explicit_slug:
        slug = explicit_slug
        logger.bind(case_id=str(case_id), template_slug=slug, doc_count=len(doc_ids)).info(
            "classify_doc_set_explicit"
        )
    else:
        counts: Counter[str] = Counter(doc_types)
        majority = counts.most_common(1)[0][0] if counts else "commercial_contract"
        slug = "nda" if majority == "nda" else "commercial_contract"
        logger.bind(
            case_id=str(case_id),
            doc_type_counts=dict(counts),
            chosen_slug=slug,
        ).info("classify_doc_set_voted")

    return {
        "document_ids": doc_ids,
        "template_slug": slug,
    }
