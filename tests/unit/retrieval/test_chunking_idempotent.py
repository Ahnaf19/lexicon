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
            # Extract inserted values from the INSERT statement's compile params
            # The rows are passed via insert().values([...])
            compiled = stmt.compile()
            # Walk the INSERT's VALUES and record them
            if hasattr(stmt, "_values") and stmt._values:
                for row_vals in stmt._values:  # type: ignore[attr-defined]
                    self.inserted_rows.append({k.key: v.value for k, v in row_vals.items()})
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
    """Two calls produce windows with identical sorted text sets."""
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
    assert texts1 == texts2


def _window_texts(session: _FakeChunkSession) -> list[str]:
    return [
        r.get("text", "") if isinstance(r, dict) else getattr(r, "text", "")
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
