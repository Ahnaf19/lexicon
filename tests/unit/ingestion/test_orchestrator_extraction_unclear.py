"""Extraction failure path: status=extraction_unclear, partial meta, no exception."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import FakeSessionFactory, make_block, make_marker_output, make_meta


@pytest.mark.asyncio
async def test_extraction_unclear_no_raise(fake_sf: FakeSessionFactory) -> None:
    blocks = [make_block("Some contract text.", confidence=0.9)]
    marker_out = make_marker_output(blocks)

    # Simulate exhausted retries → status "extraction_unclear" with partial meta
    partial_meta = make_meta(doc_type="other")
    partial_meta = partial_meta.model_copy(update={"confidence": 0.0, "parties": []})

    with (
        patch("app.ingestion.orchestrator.run_marker", new_callable=AsyncMock, return_value=marker_out),
        patch("app.ingestion.orchestrator.re_ocr_blocks", new_callable=AsyncMock, return_value=[]),
        patch(
            "app.ingestion.orchestrator.extract_document_meta",
            new_callable=AsyncMock,
            return_value=(partial_meta, "extraction_unclear"),
        ),
        patch("app.ingestion.orchestrator.deskew_pdf_bytes", side_effect=lambda b: b),
    ):
        from app.ingestion.orchestrator import ingest_document

        # Must not raise
        doc_id = await ingest_document(
            file_bytes=b"%PDF-fake",
            filename="unclear.pdf",
            mime="application/pdf",
            case_id=uuid.UUID(int=0),
            session_factory=fake_sf,
        )

    assert doc_id is not None

    # Find the Document row
    all_docs: list = []
    for s in fake_sf.sessions:
        all_docs.extend(s.added.get("Document", []))

    assert len(all_docs) == 1
    doc_row = all_docs[0]
    assert doc_row.status == "extraction_unclear"
    # Partial meta must be persisted (doc_type from partial)
    assert doc_row.doc_type == "other"
