"""validate_item node — enforces anti-hallucination invariants V1–V5 (PRD §5e)."""

from __future__ import annotations

import re
import uuid

from loguru import logger

from app.generation.state import ChecklistState, DraftChecklistItem
from app.generation.templates.types import TemplateItem
from app.models.pydantic_models import ChecklistItem, EvidenceCitation
from app.retrieval.hybrid_search import SearchHit

_PAGE_REF_RE = re.compile(r"\[doc=([^\s\]]+)\s+p\.(\d+)\]")
_Ei_RE = re.compile(r"^E(\d+)$", re.IGNORECASE)

_VALID_STATUSES = {"present", "missing", "unclear"}


def _resolve_ei(label: str, hits: list[SearchHit]) -> SearchHit | None:
    """Return the SearchHit for label 'Ei' (1-based); None if out-of-range or malformed."""
    m = _Ei_RE.match(label.strip())
    if not m:
        return None
    idx = int(m.group(1)) - 1  # convert to 0-based
    if 0 <= idx < len(hits):
        return hits[idx]
    return None


def validate_and_build(
    draft: DraftChecklistItem,
    template_item: TemplateItem,
    template_id: uuid.UUID,
    hits: list[SearchHit],
    item_slug: str,
) -> ChecklistItem:
    """Apply V1–V5 and build a final ChecklistItem from a draft."""
    status = draft.status
    rationale = draft.rationale
    cited_labels = list(draft.cited_evidence)

    # V4: Invalid status coercion (check first so downstream logic is on clean values)
    if status not in _VALID_STATUSES:
        logger.bind(item_slug=item_slug, bad_status=status).warning("validate_V4_bad_status")
        status = "unclear"
        rationale = "Invalid status returned by LLM."
        cited_labels = []

    # V1: present with no citations → unclear
    if status == "present" and not cited_labels:
        logger.bind(item_slug=item_slug).warning("validate_V1_no_citations")
        status = "unclear"
        rationale = "No supporting evidence found."

    # V2: validate each Ei label; drop mismatches / out-of-range
    retained_hits: list[SearchHit] = []
    for label in cited_labels:
        hit = _resolve_ei(label, hits)
        if hit is None:
            logger.bind(item_slug=item_slug, label=label).warning("validate_V2_label_out_of_range")
            continue
        retained_hits.append(hit)

    # V3: present after V2 but all citations dropped → re-apply V1
    if status == "present" and not retained_hits:
        logger.bind(item_slug=item_slug).warning("validate_V3_all_citations_dropped")
        status = "unclear"
        rationale = "No supporting evidence found."

    # V5: check [doc=X p.N] references in rationale against retained hits
    retained_keys = {
        (str(h.doc_id), str(h.page_number)) for h in retained_hits
    }
    unverified = False
    for m in _PAGE_REF_RE.finditer(rationale):
        doc_ref, page_ref = m.group(1), m.group(2)
        if (doc_ref, page_ref) not in retained_keys:
            unverified = True
            break
    if unverified:
        rationale = rationale + " (Note: page reference in rationale could not be verified)"
        logger.bind(item_slug=item_slug).warning("validate_V5_unverified_page_ref")

    evidence: list[EvidenceCitation] = [h.to_evidence_citation() for h in retained_hits]

    return ChecklistItem(
        id=uuid.uuid4(),
        source_template_item_id=template_item.stable_uuid(template_id),
        category=template_item.category,
        title=template_item.title,
        description=template_item.description,
        status=status,
        required=template_item.required,
        evidence=evidence,
        confidence=draft.confidence,
        rationale=rationale,
        learned_from_pattern_ids=[],
    )


async def validate_item(state: ChecklistState) -> dict[str, object]:
    """Resolve Ei citations and enforce invariants for the current item."""
    slug = state["current_item_slug"]
    assert slug is not None

    in_progress = dict(state.get("items_in_progress") or {})
    entry = in_progress.get(slug)

    hits = (state.get("search_hits_by_item") or {}).get(slug, [])
    template = state["template"]
    template_item = next(i for i in template.items if i.slug == slug)

    # Entry is a sentinel dict from draft_item or None (shouldn't happen).
    if not isinstance(entry, dict) or "_draft" not in entry:
        # Safety valve: create an unclear item if draft_item failed silently.
        draft = DraftChecklistItem(
            status="unclear",
            confidence=0.0,
            rationale="Draft node produced no output.",
            cited_evidence=[],
        )
    else:
        draft = entry["_draft"]

    checklist_item = validate_and_build(
        draft=draft,
        template_item=template_item,
        template_id=template.id,
        hits=hits,
        item_slug=slug,
    )

    logger.bind(
        item_slug=slug,
        status=checklist_item.status,
        evidence_count=len(checklist_item.evidence),
    ).info("validate_item_ok")

    in_progress[slug] = checklist_item
    return {"items_in_progress": in_progress}
