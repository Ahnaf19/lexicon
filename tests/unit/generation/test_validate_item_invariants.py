"""Tests for anti-hallucination invariants V1–V5 in validate_item (PRD §5e)."""

from __future__ import annotations

import uuid

import pytest

from app.generation.nodes.validate_item import validate_and_build
from app.generation.state import DraftChecklistItem
from app.generation.templates.commercial_contract import COMMERCIAL_CONTRACT
from app.models.pydantic_models import EvidenceCitation
from app.retrieval.hybrid_search import SearchHit

_TEMPLATE = COMMERCIAL_CONTRACT
_PARTIES = next(i for i in _TEMPLATE.items if i.slug == "parties")


def _hit(doc_id: uuid.UUID | None = None, page: int = 1, parent_id: uuid.UUID | None = None) -> SearchHit:
    d = doc_id or uuid.uuid4()
    return SearchHit(
        citation_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        doc_id=d,
        page_number=page,
        char_offset_start=0,
        char_offset_end=100,
        snippet="Some snippet text here.",
        context_text="Full parent section text with legal content.",
        retrieval_score=0.8,
    )


def _build(
    status: str,
    cited_evidence: list[str],
    rationale: str = "Some rationale.",
    confidence: float = 0.9,
    hits: list[SearchHit] | None = None,
):
    draft = DraftChecklistItem(
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=rationale,
        cited_evidence=cited_evidence,
    )
    return validate_and_build(
        draft=draft,
        template_item=_PARTIES,
        template_id=_TEMPLATE.id,
        hits=hits or [],
        item_slug=_PARTIES.slug,
    )


# ---------------------------------------------------------------------------
# V1: present with no citations → unclear
# ---------------------------------------------------------------------------


def test_V1_present_no_citations_becomes_unclear():
    result = _build(status="present", cited_evidence=[], hits=[_hit()])
    assert result.status == "unclear"
    assert result.evidence == []
    assert "No supporting evidence" in result.rationale


# ---------------------------------------------------------------------------
# V2: out-of-range or malformed Ei label → dropped
# ---------------------------------------------------------------------------


def test_V2_out_of_range_label_dropped():
    h = _hit()
    # Only 1 hit, but model claims E2 (index 2, 0-based index 1 = out of range).
    result = _build(status="present", cited_evidence=["E2"], hits=[h])
    # After V2 all cites dropped, V3 re-applies V1 → unclear
    assert result.status == "unclear"
    assert result.evidence == []


def test_V2_valid_label_retained():
    h = _hit()
    result = _build(status="present", cited_evidence=["E1"], hits=[h])
    assert result.status == "present"
    assert len(result.evidence) == 1
    assert result.evidence[0].chunk_id == h.chunk_id


# ---------------------------------------------------------------------------
# V3: all citations dropped after V2 → re-coerce to unclear
# ---------------------------------------------------------------------------


def test_V3_all_citations_dropped_becomes_unclear():
    h = _hit()
    # E3 is out-of-range (only 1 hit), so V2 drops it; V3 triggers.
    result = _build(status="present", cited_evidence=["E3"], hits=[h])
    assert result.status == "unclear"
    assert result.evidence == []


# ---------------------------------------------------------------------------
# V4: invalid status → unclear
# ---------------------------------------------------------------------------


def test_V4_invalid_status_coerced():
    # DraftChecklistItem validation prevents invalid status, so call validate_and_build directly.
    draft = DraftChecklistItem(
        status="unclear",  # will be mutated below via dict bypass
        confidence=0.5,
        rationale="bad",
        cited_evidence=[],
    )
    # Simulate model returning a bad status by constructing with model_construct.
    draft_bad = DraftChecklistItem.model_construct(
        status="HALLUCINATED",  # type: ignore[arg-type]
        confidence=0.5,
        rationale="bad status",
        cited_evidence=[],
    )
    result = validate_and_build(
        draft=draft_bad,
        template_item=_PARTIES,
        template_id=_TEMPLATE.id,
        hits=[],
        item_slug=_PARTIES.slug,
    )
    assert result.status == "unclear"
    assert "Invalid status" in result.rationale


# ---------------------------------------------------------------------------
# V5: unverified page reference in rationale → note appended
# ---------------------------------------------------------------------------


def test_V5_unverified_page_ref_flagged():
    doc_id = uuid.uuid4()
    h = _hit(doc_id=doc_id, page=3)
    # Rationale references page 99, which is not in the evidence.
    rationale = f"[doc={doc_id} p.99] something stated here."
    result = _build(
        status="present",
        cited_evidence=["E1"],
        rationale=rationale,
        hits=[h],
    )
    assert "could not be verified" in result.rationale


def test_V5_verified_page_ref_no_note():
    doc_id = uuid.uuid4()
    h = _hit(doc_id=doc_id, page=5)
    rationale = f"[doc={doc_id} p.5] parties clearly identified."
    result = _build(
        status="present",
        cited_evidence=["E1"],
        rationale=rationale,
        hits=[h],
    )
    assert "could not be verified" not in result.rationale


# ---------------------------------------------------------------------------
# status="missing" — no citations required; should pass V1 unmodified
# ---------------------------------------------------------------------------


def test_missing_status_with_no_citations_passes():
    """status='missing' with empty cited_evidence must NOT be coerced by V1."""
    result = _build(status="missing", cited_evidence=[], hits=[_hit()])
    assert result.status == "missing"
    assert result.evidence == []
    # Rationale must be preserved, not replaced by the V1 message.
    assert "No supporting evidence" not in result.rationale


# ---------------------------------------------------------------------------
# V5: both [doc=X p.N] and [doc=X page=N] formats handled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ref_format", ["p.{p}", "page={p}"])
def test_V5_verified_page_ref_both_formats(ref_format: str) -> None:
    doc_id = uuid.uuid4()
    h = _hit(doc_id=doc_id, page=5)
    rationale = f"[doc={doc_id} {ref_format.format(p=5)}] parties clearly identified."
    result = _build(status="present", cited_evidence=["E1"], rationale=rationale, hits=[h])
    assert "could not be verified" not in result.rationale


@pytest.mark.parametrize("ref_format", ["p.{p}", "page={p}"])
def test_V5_unverified_page_ref_both_formats(ref_format: str) -> None:
    doc_id = uuid.uuid4()
    h = _hit(doc_id=doc_id, page=3)
    rationale = f"[doc={doc_id} {ref_format.format(p=99)}] page not in retained evidence."
    result = _build(status="present", cited_evidence=["E1"], rationale=rationale, hits=[h])
    assert "could not be verified" in result.rationale
