"""Tests for expand_with_parents: budget packing (continue not break) and context_text field."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.retrieval.hybrid_search import PARENT_BUDGET_TOKENS, SearchHit, expand_with_parents
from app.retrieval.chunking import _ENCODER


def _make_row(
    chunk_id: uuid.UUID,
    doc_id: uuid.UUID,
    text: str,
    parent_section_id: uuid.UUID | None,
    rrf_score: float = 1.0,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "page_number": 1,
        "char_offset_start": 0,
        "char_offset_end": len(text),
        "text": text,
        "parent_section_id": parent_section_id,
        "rrf_score": rrf_score,
        "dense_rank": 1,
        "sparse_rank": 1,
    }


class _FakeSearchSession:
    def __init__(self, parent_rows: list[tuple[uuid.UUID, str]]) -> None:
        self._parent_rows = parent_rows

    async def execute(self, stmt: Any, params: Any = None) -> Any:
        result = MagicMock()
        result.__iter__ = MagicMock(return_value=iter(self._parent_rows))
        return result


@pytest.mark.asyncio
async def test_oversized_parent_skipped_not_breaks() -> None:
    """A mid-ranked oversized parent is skipped; lower-ranked small parent is still included.

    Scenario (3 rows):
      rank-1: small parent  → always included (first-hit rule), tokens_used = small
      rank-2: huge parent   → skipped (tokens_used + huge > budget, hits non-empty)
      rank-3: small parent  → included with continue, NOT included with break

    The `and hits` guard means rank-1 is always included. The distinction between
    continue and break matters for rank-3 when rank-2 is oversized.
    """
    doc_id = uuid.uuid4()
    parent_id_1 = uuid.uuid4()
    parent_id_2 = uuid.uuid4()
    parent_id_3 = uuid.uuid4()

    # rank-1: small → fits, sets tokens_used to a small number
    small_parent_1 = "Small parent text that fits easily in the budget."
    # rank-2: huge → causes skip (tokens_used + huge > budget)
    huge_parent = "word " * (PARENT_BUDGET_TOKENS + 500)
    # rank-3: small → should fit since skip (not break) leaves remaining budget
    small_parent_3 = "Another small parent text for rank-3."

    row1 = _make_row(uuid.uuid4(), doc_id, "Window text 1", parent_id_1, rrf_score=0.9)
    row2 = _make_row(uuid.uuid4(), doc_id, "Window text 2", parent_id_2, rrf_score=0.7)
    row3 = _make_row(uuid.uuid4(), doc_id, "Window text 3", parent_id_3, rrf_score=0.5)

    session = _FakeSearchSession(
        [(parent_id_1, small_parent_1), (parent_id_2, huge_parent), (parent_id_3, small_parent_3)]
    )

    hits = await expand_with_parents([row1, row2, row3], session)

    chunk_ids = {h.chunk_id for h in hits}
    assert row1["chunk_id"] in chunk_ids, "rank-1 (small) must be included"
    assert row2["chunk_id"] not in chunk_ids, "rank-2 (huge) must be skipped"
    assert row3["chunk_id"] in chunk_ids, (
        "rank-3 (small) must be included — continue skips oversized, break would stop here"
    )


@pytest.mark.asyncio
async def test_first_hit_always_included() -> None:
    """The very first hit is always included even if it exceeds the budget (and hits is empty)."""
    doc_id = uuid.uuid4()
    parent_id = uuid.uuid4()

    huge_parent = "word " * (PARENT_BUDGET_TOKENS + 500)
    row = _make_row(uuid.uuid4(), doc_id, "Window text", parent_id, rrf_score=1.0)

    session = _FakeSearchSession([(parent_id, huge_parent)])

    hits = await expand_with_parents([row], session)

    assert len(hits) == 1, "First hit must always be included regardless of budget"
    assert hits[0].chunk_id == row["chunk_id"]


@pytest.mark.asyncio
async def test_context_text_uses_parent_when_available() -> None:
    """context_text is the parent section text when parent_section_id is set."""
    doc_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    parent_text = "PARENT SECTION FULL TEXT"
    window_text = "Window text preview."

    row = _make_row(uuid.uuid4(), doc_id, window_text, parent_id)
    session = _FakeSearchSession([(parent_id, parent_text)])

    hits = await expand_with_parents([row], session)

    assert len(hits) == 1
    assert hits[0].context_text == parent_text
    assert hits[0].snippet == window_text[:300]


@pytest.mark.asyncio
async def test_context_text_falls_back_to_window_when_no_parent() -> None:
    """context_text equals window text when parent_section_id is None."""
    doc_id = uuid.uuid4()
    window_text = "Window text with no parent section."

    row = _make_row(uuid.uuid4(), doc_id, window_text, parent_section_id=None)
    session = _FakeSearchSession([])

    hits = await expand_with_parents([row], session)

    assert len(hits) == 1
    assert hits[0].context_text == window_text
    assert hits[0].snippet == window_text[:300]
