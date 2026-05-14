"""Few-shot example bank — CIPHER-pattern retrieval of operator edit pairs (PRD §5f Layer 1/3).

On finalize, every edited item produces one row in few_shot_examples (written by
pattern_extractor).  At draft_item time, retrieve_few_shot returns the top-k most similar
(original_draft, final_item) pairs for injection into the generation prompt.
"""

from __future__ import annotations

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.templates.types import TemplateItem
from app.models.pydantic_models import ChecklistItem
from app.retrieval.embedding import embed_query


async def retrieve_few_shot(
    session: AsyncSession,
    template_item: TemplateItem,
    doc_type: str,
    evidence_summary: str,
    k: int = 3,
) -> list[tuple[ChecklistItem, ChecklistItem]]:
    """Return up to k (original_draft, final_item) pairs ranked by embedding similarity.

    Pairs where original_draft == final_item are excluded — they teach nothing (P3).
    Filtered to the given doc_type so cross-domain noise is avoided.
    """
    query_text = (
        template_item.title
        + " "
        + template_item.category
        + " "
        + evidence_summary[:500]
    )
    embedding = await embed_query(query_text)

    # Cosine ordering via pgvector <=> operator; LIMIT k for efficiency (L2).
    rows = await session.execute(
        text(
            """
            SELECT original_draft, final_item
            FROM few_shot_examples
            WHERE doc_type = :doc_type
            ORDER BY context_embedding <=> CAST(:emb AS vector)
            LIMIT :k
            """
        ),
        {
            "doc_type": doc_type,
            "emb": str(embedding),
            "k": k,
        },
    )

    pairs: list[tuple[ChecklistItem, ChecklistItem]] = []
    for original_dict, final_dict in rows:
        try:
            original = ChecklistItem.model_validate(original_dict)
            final = ChecklistItem.model_validate(final_dict)
        except Exception as exc:
            logger.bind(doc_type=doc_type, error=str(exc)[:120]).warning(
                "few_shot_deserialize_failed"
            )
            continue

        # P3 defensive filter: skip identity pairs even if they slipped into the DB.
        if original.model_dump() == final.model_dump():
            continue

        pairs.append((original, final))

    logger.bind(
        doc_type=doc_type,
        template_item_slug=template_item.slug,
        retrieved=len(pairs),
    ).debug("few_shot_retrieved")
    return pairs
