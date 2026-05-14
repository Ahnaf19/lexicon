"""rechunk_document is idempotent: calling twice yields same row count and text."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.sqlalchemy_models import Chunk


def _make_chunk(text: str, page: int = 1, offset_start: int = 0) -> Chunk:
    c = Chunk()
    c.id = uuid.uuid4()
    c.doc_id = uuid.uuid4()
    c.page_number = page
    c.section_heading = None
    c.char_offset_start = offset_start
    c.char_offset_end = offset_start + len(text)
    c.text = text
    c.embedding = None
    c.parent_section_id = None
    c.meta = {
        "bbox": {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 20.0},
        "ocr_confidence": 0.9,
        "ocr_engine": "marker",
        "is_handwriting": False,
        "low_ocr_confidence": False,
    }
    return c


class _FakeChunkSession:
    """FakeSession that returns canned Chunk rows for SELECT and records inserts."""

    def __init__(self, source_chunks: list[Chunk]) -> None:
        self._source_chunks = source_chunks
        self.inserted_rows: list[dict[str, Any]] = []
        self._committed = False

    async def execute(self, stmt: Any, params: Any = None) -> Any:
        stmt_str = str(stmt)
        if "INSERT" in stmt_str.upper():
            # SQLAlchemy 2.x: insert().values([dict, ...]) stores rows in _multi_values
            # as tuple[list[dict[Column, value]]]. Column keys must be accessed via col.key.
            if hasattr(stmt, "_multi_values") and stmt._multi_values:
                for row_group in stmt._multi_values:  # type: ignore[attr-defined]
                    for row in row_group:
                        self.inserted_rows.append({col.key: val for col, val in row.items()})
            return MagicMock()
        if "DELETE" in stmt_str.upper():
            return MagicMock()
        # SELECT — return all source chunks
        result = MagicMock()
        result.scalars.return_value.all.return_value = self._source_chunks
        return result

    async def get(self, model: Any, pk: Any) -> Any:
        return None

    async def commit(self) -> None:
        self._committed = True

    async def __aenter__(self) -> "_FakeChunkSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_rechunk_idempotent_row_count() -> None:
    """Two calls to rechunk_document produce the same window count."""
    doc_id = uuid.uuid4()
    text = (
        "# Definitions\n"
        "As used herein, 'Confidential Information' means any data shared.\n\n"
        "## Obligations\n"
        "Each party shall protect the other party's Confidential Information.\n"
    )
    source_chunks = [_make_chunk(text)]
    for c in source_chunks:
        c.doc_id = doc_id

    session1 = _FakeChunkSession(source_chunks)
    wc1, sc1 = await _rechunk(doc_id, session1)

    session2 = _FakeChunkSession(source_chunks)
    wc2, sc2 = await _rechunk(doc_id, session2)

    assert wc1 == wc2, f"Window count changed: {wc1} vs {wc2}"
    assert sc1 == sc2, f"Section count changed: {sc1} vs {sc2}"


@pytest.mark.asyncio
async def test_rechunk_idempotent_window_texts() -> None:
    """Two calls produce windows with identical sorted text sets and char offsets."""
    doc_id = uuid.uuid4()
    text = "# Agreement\nThis NDA governs the sharing of information.\n" * 3

    source_chunks = [_make_chunk(text)]
    for c in source_chunks:
        c.doc_id = doc_id

    session1 = _FakeChunkSession(source_chunks)
    await _rechunk(doc_id, session1)

    session2 = _FakeChunkSession(source_chunks)
    await _rechunk(doc_id, session2)

    texts1 = sorted(_window_texts(session1))
    texts2 = sorted(_window_texts(session2))

    assert len(texts1) > 0, "Expected at least one window — inserted_rows may be empty (fake bug)"
    assert texts1 == texts2

    offsets1 = sorted(_window_offsets(session1))
    offsets2 = sorted(_window_offsets(session2))
    assert offsets1 == offsets2, "char_offset_start values must be stable across re-runs"


def _window_texts(session: _FakeChunkSession) -> list[str]:
    return [
        r.get("text", "") if isinstance(r, dict) else getattr(r, "text", "")
        for r in session.inserted_rows
        if _is_window(r)
    ]


def _window_offsets(session: _FakeChunkSession) -> list[int]:
    return [
        r.get("char_offset_start", -1) if isinstance(r, dict) else getattr(r, "char_offset_start", -1)
        for r in session.inserted_rows
        if _is_window(r)
    ]


def _is_window(row: Any) -> bool:
    if isinstance(row, dict):
        meta = row.get("meta", {})
    else:
        meta = getattr(row, "meta", {}) or {}
    return meta.get("kind") == "window"


async def _rechunk(doc_id: uuid.UUID, session: _FakeChunkSession) -> tuple[int, int]:
    from app.retrieval.chunking import rechunk_document

    return await rechunk_document(doc_id, session)  # type: ignore[arg-type]
