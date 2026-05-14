"""retrieve_evidence node — hybrid search for the current template item, deduped by parent."""

from __future__ import annotations

from loguru import logger

from app.core.db import SessionLocal
from app.generation.state import ChecklistState
from app.retrieval.hybrid_search import SearchHit, search


async def retrieve_evidence(state: ChecklistState) -> dict[str, object]:
    """Retrieve and dedupe SearchHits for the current item (G7)."""
    template = state["template"]
    slug = state["current_item_slug"]
    assert slug is not None

    item = next(i for i in template.items if i.slug == slug)
    case_id = state["case_id"]

    async with SessionLocal() as session:
        hits = await search(item.sub_query, case_id, session, k=8)

    # Dedupe by context_text: two windows from the same parent section share identical
    # context_text and would waste context budget if both were sent to the LLM (G7).
    seen: set[str] = set()
    deduped: list[SearchHit] = []
    for h in hits:
        if h.context_text not in seen:
            seen.add(h.context_text)
            deduped.append(h)

    logger.bind(
        item_slug=slug,
        original_count=len(hits),
        deduped_count=len(deduped),
    ).debug("retrieve_evidence_deduped")

    hits_by_item = dict(state.get("search_hits_by_item") or {})
    hits_by_item[slug] = deduped
    return {"search_hits_by_item": hits_by_item}
