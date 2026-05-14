"""Image input: wrap_image_as_pdf is called; original_mime recorded; pipeline completes."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import FakeSessionFactory, make_block, make_marker_output, make_meta


@pytest.mark.asyncio
async def test_jpg_is_wrapped_as_pdf(fake_sf: FakeSessionFactory) -> None:
    fake_pdf_bytes = b"%PDF-1.4-wrapped"
    blocks = [make_block("handwritten exhibit text", confidence=0.6, is_handwriting=True)]
    marker_out = make_marker_output(blocks)
    doc_meta = make_meta()

    with (
        patch(
            "app.ingestion.orchestrator.wrap_image_as_pdf",
            return_value=fake_pdf_bytes,
        ) as mock_wrap,
        patch("app.ingestion.orchestrator.run_marker", new_callable=AsyncMock, return_value=marker_out),
        patch(
            "app.ingestion.orchestrator.re_ocr_blocks",
            new_callable=AsyncMock,
            return_value=[("better handwriting", 0.85)],
        ),
        patch("app.ingestion.orchestrator.extract_document_meta", new_callable=AsyncMock, return_value=(doc_meta, "ok")),
        patch("app.ingestion.orchestrator.deskew_pdf_bytes", side_effect=lambda b: b),
    ):
        from app.ingestion.orchestrator import ingest_document

        doc_id = await ingest_document(
            file_bytes=b"\xff\xd8\xff-fake-jpeg",
            filename="exhibit_b.jpg",
            mime="image/jpeg",
            case_id=uuid.UUID(int=0),
            session_factory=fake_sf,
        )

    mock_wrap.assert_called_once()

    # original_mime should be recorded in meta
    all_docs: list = []
    for s in fake_sf.sessions:
        all_docs.extend(s.added.get("Document", []))

    assert len(all_docs) == 1
    doc_row = all_docs[0]
    assert doc_row.meta.get("original_mime") == "image/jpeg"
    assert doc_id is not None
