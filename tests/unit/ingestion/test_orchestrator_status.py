"""Assert orchestrator never sets status='indexed' directly on the success path.

index_document() is responsible for the 'indexed' promotion after embedding.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import FakeSessionFactory, make_block, make_marker_output, make_meta


@pytest.mark.asyncio
async def test_success_path_status_is_ocr_done_not_indexed(
    fake_sf: FakeSessionFactory,
) -> None:
    """On a fully successful ingest, final status must be 'ocr_done', never 'indexed'."""
    blocks = [make_block("Agreement text.", confidence=0.95)]
    marker_out = make_marker_output(blocks)
    doc_meta = make_meta()

    with (
        patch("app.ingestion.orchestrator.run_marker", new_callable=AsyncMock, return_value=marker_out),
        patch("app.ingestion.orchestrator.re_ocr_blocks", new_callable=AsyncMock),
        patch(
            "app.ingestion.orchestrator.extract_document_meta",
            new_callable=AsyncMock,
            return_value=(doc_meta, "ok"),
        ),
        patch("app.ingestion.orchestrator.deskew_pdf_bytes", side_effect=lambda b: b),
    ):
        from app.ingestion.orchestrator import ingest_document

        await ingest_document(
            file_bytes=b"%PDF-fake",
            filename="contract.pdf",
            mime="application/pdf",
            case_id=uuid.UUID(int=0),
            session_factory=fake_sf,
        )

    all_docs: list = []
    for s in fake_sf.sessions:
        all_docs.extend(s.added.get("Document", []))

    assert len(all_docs) == 1
    doc = all_docs[0]
    assert doc.status == "ocr_done", (
        f"Orchestrator must not set status='indexed' directly; got '{doc.status}'"
    )
    assert doc.status != "indexed"


@pytest.mark.asyncio
async def test_extraction_unclear_status_unchanged(fake_sf: FakeSessionFactory) -> None:
    """extraction_unclear path must still set that status, not ocr_done."""
    blocks = [make_block("Unclear text.", confidence=0.9)]
    marker_out = make_marker_output(blocks)
    partial_meta = make_meta()

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

        await ingest_document(
            file_bytes=b"%PDF-fake",
            filename="unclear.pdf",
            mime="application/pdf",
            case_id=uuid.UUID(int=0),
            session_factory=fake_sf,
        )

    all_docs: list = []
    for s in fake_sf.sessions:
        all_docs.extend(s.added.get("Document", []))

    doc = all_docs[0]
    assert doc.status == "extraction_unclear"
