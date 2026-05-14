"""Hybrid search integration with pure in-process fakes (no real DB)."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.retrieval.hybrid_search import SearchHit, rrf_merge, search


_DOC_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_CASE_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_SECTION_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
_WINDOW_ID_1 = uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
_WINDOW_ID_2 = uuid.UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")

_WINDOW_ROW_1 = {
    "chunk_id": _WINDOW_ID_1,
    "doc_id": _DOC_ID,
    "page_number": 1,
    "char_offset_start": 0,
    "char_offset_end": 200,
    "text": "Confidential information means any data marked confidential.",
    "parent_section_id": _SECTION_ID,
    "score": 0.92,
}

_WINDOW_ROW_2 = {
    "chunk_id": _WINDOW_ID_2,
    "doc_id": _DOC_ID,
    "page_number": 2,
    "char_offset_start": 200,
    "char_offset_end": 400,
    "text": "The parties agree to hold information in strict confidence.",
    "parent_section_id": _SECTION_ID,
    "score": 0.85,
}

_SECTION_TEXT = (
    "# Confidentiality\n"
    "Confidential information means any data marked confidential. "
    "The parties agree to hold information in strict confidence."
)


class _FakeSearchSession:
    """Returns canned rows for dense/sparse SQL and parent section rows."""

    def __init__(self) -> None:
        self._parent_fetched = False

    async def execute(self, stmt: Any, params: Any = None) -> Any:
        result = MagicMock()

        stmt_str = str(stmt) if hasattr(stmt, "__str__") else ""

        if "SELECT id, text FROM chunks" in stmt_str:
            # Parent expansion query
            rows = [(_SECTION_ID, _SECTION_TEXT)]
            result.__iter__ = lambda s: iter(rows)
            return result

        # For dense/sparse — return window rows via mappings
        rows = [_make_row_mapping(_WINDOW_ROW_1), _make_row_mapping(_WINDOW_ROW_2)]
        result.__iter__ = lambda s: iter(rows)
        return result

    async def __aenter__(self) -> "_FakeSearchSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


def _make_row_mapping(d: dict) -> Any:
    m = MagicMock()
    m._mapping = d
    return m


@pytest.mark.asyncio
async def test_rrf_merge_shape_from_canned_branches() -> None:
    """RRF merge over canned dense+sparse lists yields SearchHit-compatible dicts."""
    dense_rows = [_WINDOW_ROW_1, _WINDOW_ROW_2]
    sparse_rows = [_WINDOW_ROW_2, _WINDOW_ROW_1]

    merged = rrf_merge(dense_rows, sparse_rows)

    assert len(merged) >= 2
    assert "rrf_score" in merged[0]
    assert "dense_rank" in merged[0]
    assert "sparse_rank" in merged[0]
    # Both chunks present
    chunk_ids = {r["chunk_id"] for r in merged}
    assert _WINDOW_ID_1 in chunk_ids
    assert _WINDOW_ID_2 in chunk_ids


@pytest.mark.asyncio
async def test_search_returns_evidence_citation_shape() -> None:
    """search() returns list[SearchHit] with valid EvidenceCitation fields."""
    fake_session = _FakeSearchSession()

    with patch(
        "app.retrieval.hybrid_search.embed_query",
        new_callable=AsyncMock,
        return_value=[0.1] * 768,
    ):
        hits = await search("confidential information", _CASE_ID, fake_session)  # type: ignore[arg-type]

    assert isinstance(hits, list)
    assert len(hits) > 0
    for hit in hits:
        assert isinstance(hit, SearchHit)
        assert isinstance(hit.chunk_id, uuid.UUID)
        assert isinstance(hit.doc_id, uuid.UUID)
        assert isinstance(hit.page_number, int)
        assert isinstance(hit.snippet, str)
        assert isinstance(hit.retrieval_score, float)
        # to_evidence_citation() must succeed
        citation = hit.to_evidence_citation()
        assert citation.chunk_id == hit.chunk_id


@pytest.mark.asyncio
async def test_search_empty_results() -> None:
    """search() returns empty list when no window chunks match."""

    class _EmptySession:
        async def execute(self, stmt: Any, params: Any = None) -> Any:
            result = MagicMock()
            result.__iter__ = lambda s: iter([])
            return result

        async def __aenter__(self) -> "_EmptySession":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    with patch(
        "app.retrieval.hybrid_search.embed_query",
        new_callable=AsyncMock,
        return_value=[0.1] * 768,
    ):
        hits = await search("no match", _CASE_ID, _EmptySession())  # type: ignore[arg-type]

    assert hits == []
