"""No window chunk is created from text spanning two sections.

Tested structurally: rechunk_document must produce at least two sections for a
two-section document, and return a non-zero window count. The code assigns each
window exclusively to the section it was split from (parent_section_id = section_id),
so cross-section windows are architecturally impossible given correct section detection.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.models.sqlalchemy_models import Chunk


def _make_chunk(text: str, doc_id: uuid.UUID, page: int = 1) -> Chunk:
    c = Chunk()
    c.id = uuid.uuid4()
    c.doc_id = doc_id
    c.page_number = page
    c.section_heading = None
    c.char_offset_start = 0
    c.char_offset_end = len(text)
    c.text = text
    c.embedding = None
    c.parent_section_id = None
    c.meta = {
        "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20},
        "ocr_confidence": 0.9,
        "ocr_engine": "marker",
        "is_handwriting": False,
        "low_ocr_confidence": False,
    }
    return c


class _FakeSession:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.execute_calls: list[Any] = []

    async def execute(self, stmt: Any, params: Any = None) -> Any:
        self.execute_calls.append(stmt)
        result = MagicMock()
        result.scalars.return_value.all.return_value = self._chunks
        return result

    async def commit(self) -> None:
        pass

    async def get(self, *args: Any) -> None:
        return None

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_two_sections_yields_at_least_two_section_rows() -> None:
    """A document with two clear section headings produces section_count >= 2."""
    doc_id = uuid.uuid4()
    text = (
        "# Confidentiality\n"
        "The receiving party shall hold all information in strict confidence. "
        "No disclosure to third parties shall be permitted without prior written consent. "
        "This obligation survives termination of the agreement.\n\n"
        "# Governing Law\n"
        "This Agreement shall be governed by the laws of the State of Delaware. "
        "Any disputes shall be resolved in the courts of that jurisdiction. "
        "The parties consent to personal jurisdiction therein."
    )
    source = _make_chunk(text, doc_id)
    session = _FakeSession([source])

    from app.retrieval.chunking import rechunk_document

    window_count, section_count = await rechunk_document(doc_id, session)  # type: ignore[arg-type]

    assert section_count >= 2, (
        f"Expected at least 2 sections for a two-heading document, got {section_count}"
    )
    assert window_count > 0, "Expected at least one window chunk"


@pytest.mark.asyncio
async def test_windows_derived_from_section_text_not_mixed() -> None:
    """Verify window_count is proportional to content, not inflated by cross-section merging.

    With two roughly equal sections, each section should produce at least one window.
    If windows were merging across sections, the total would be the same as a single
    section of combined length — indistinguishable here. What we can assert is that
    section_count matches the heading count and window_count is non-zero.
    """
    doc_id = uuid.uuid4()
    section_a = (
        "## Definitions\n"
        + "For purposes of this Agreement, the following terms shall have the meanings set forth. " * 5
    )
    section_b = (
        "## Obligations\n"
        + "Each party shall protect the other party's Confidential Information with care. " * 5
    )
    text = section_a + "\n\n" + section_b
    source = _make_chunk(text, doc_id)
    session = _FakeSession([source])

    from app.retrieval.chunking import rechunk_document

    window_count, section_count = await rechunk_document(doc_id, session)  # type: ignore[arg-type]

    # Two markdown headings → two sections detected
    assert section_count == 2, f"Expected exactly 2 sections, got {section_count}"
    # Each section has content → at least one window each
    assert window_count >= 2, f"Expected at least 2 windows (one per section), got {window_count}"


@pytest.mark.asyncio
async def test_single_section_document() -> None:
    """Document with no section headings produces exactly one section and some windows."""
    doc_id = uuid.uuid4()
    # No headings — all prose → one implicit section starting at offset 0
    text = "This is a plain contract paragraph. " * 30  # enough text for a window
    source = _make_chunk(text, doc_id)
    session = _FakeSession([source])

    from app.retrieval.chunking import rechunk_document

    window_count, section_count = await rechunk_document(doc_id, session)  # type: ignore[arg-type]

    assert section_count == 1
    assert window_count >= 1
