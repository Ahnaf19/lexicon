"""index_document must promote status to 'indexed' for both ocr_done and extraction_unclear docs.

Uses a focused fake session — pgvector Vector(768) prevents real SQLite from being used for
the chunks table, so we stub at the session boundary and only exercise the Python-level logic.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.sqlalchemy_models import Chunk, Document


def _make_document(status: str) -> Document:
    doc = Document()
    doc.id = uuid.uuid4()
    doc.status = status
    return doc


def _make_window_chunk(doc_id: uuid.UUID) -> Chunk:
    c = Chunk()
    c.id = uuid.uuid4()
    c.doc_id = doc_id
    c.text = "Sample window text for embedding."
    c.embedding = None
    c.meta = {"kind": "window"}
    c.page_number = 1
    c.char_offset_start = 0
    c.char_offset_end = len(c.text)
    return c


class _PromotionFakeSession:
    """Records status promotion; simulates the fast-path (pending windows > 0)."""

    def __init__(self, doc: Document, windows: list[Chunk]) -> None:
        self._doc = doc
        self._windows = windows
        self.committed = 0

    async def execute(self, stmt: Any, params: Any = None) -> Any:
        stmt_str = str(stmt).lower()
        result = MagicMock()
        if "count(" in stmt_str:
            # fast-path count query: return number of unembedded windows
            result.scalar_one.return_value = sum(
                1 for w in self._windows if w.embedding is None
            )
        elif "select" in stmt_str and "chunks" in stmt_str:
            result.all.return_value = [(w.id, w.text) for w in self._windows]
        elif "update" in stmt_str:
            for w in self._windows:
                w.embedding = [0.0] * 768
        return result

    async def get(self, model: Any, pk: uuid.UUID) -> Any:
        if model.__name__ == "Document" and pk == self._doc.id:
            return self._doc
        return None

    async def commit(self) -> None:
        self.committed += 1

    async def __aenter__(self) -> "_PromotionFakeSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


def _make_session_factory(session: _PromotionFakeSession) -> Any:
    class _Factory:
        def __call__(self) -> _PromotionFakeSession:
            return session
    return _Factory()


@pytest.mark.asyncio
async def test_ocr_done_promoted_to_indexed() -> None:
    """index_document promotes status from 'ocr_done' to 'indexed'."""
    doc = _make_document("ocr_done")
    windows = [_make_window_chunk(doc.id) for _ in range(3)]
    session = _PromotionFakeSession(doc, windows)

    zero_vecs = [[0.0] * 768 for _ in windows]
    with (
        patch("app.retrieval.indexing.embed_texts", new=AsyncMock(return_value=zero_vecs)),
        patch("app.retrieval.indexing.warmup", new=AsyncMock()),
    ):
        from app.retrieval.indexing import index_document

        await index_document(doc.id, session_factory=_make_session_factory(session))

    assert doc.status == "indexed", f"Expected 'indexed', got '{doc.status}'"
    assert session.committed >= 1


@pytest.mark.asyncio
async def test_extraction_unclear_promoted_to_indexed() -> None:
    """index_document promotes status from 'extraction_unclear' to 'indexed'."""
    doc = _make_document("extraction_unclear")
    windows = [_make_window_chunk(doc.id) for _ in range(2)]
    session = _PromotionFakeSession(doc, windows)

    zero_vecs = [[0.0] * 768 for _ in windows]
    with (
        patch("app.retrieval.indexing.embed_texts", new=AsyncMock(return_value=zero_vecs)),
        patch("app.retrieval.indexing.warmup", new=AsyncMock()),
    ):
        from app.retrieval.indexing import index_document

        await index_document(doc.id, session_factory=_make_session_factory(session))

    assert doc.status == "indexed", f"Expected 'indexed', got '{doc.status}'"
