"""TrOCR fallback: low-confidence block gets re-OCR'd; engine flips to trocr."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import FakeSessionFactory, make_block, make_marker_output, make_meta


@pytest.mark.asyncio
async def test_trocr_flips_engine_and_text(fake_sf: FakeSessionFactory) -> None:
    low_conf_block = make_block("blurry text", confidence=0.3, is_handwriting=False)
    high_conf_block = make_block("clear text", confidence=0.91, page=2)
    blocks = [low_conf_block, high_conf_block]
    marker_out = make_marker_output(blocks)
    doc_meta = make_meta()

    with (
        patch(
            "app.ingestion.orchestrator.run_marker",
            new_callable=AsyncMock,
            return_value=marker_out,
        ),
        patch(
            "app.ingestion.orchestrator.re_ocr_blocks",
            new_callable=AsyncMock,
            return_value=[("better text", 0.87)],
        ),
        patch(
            "app.ingestion.orchestrator.extract_document_meta",
            new_callable=AsyncMock,
            return_value=(doc_meta, "ok"),
        ),
        patch(
            "app.ingestion.orchestrator.deskew_pdf_bytes",
            side_effect=lambda b: b,
        ),
    ):
        from app.ingestion.orchestrator import ingest_document

        await ingest_document(
            file_bytes=b"%PDF-fake",
            filename="degraded.pdf",
            mime="application/pdf",
            case_id=uuid.UUID(int=0),
            session_factory=fake_sf,
        )

    # Find the chunk for the first block
    all_chunks: list = []
    for s in fake_sf.sessions:
        all_chunks.extend(s.added.get("Chunk", []))

    assert len(all_chunks) == 2

    # Chunk 0 should have been updated by TrOCR
    chunk0 = all_chunks[0]
    assert chunk0.text == "better text", f"Expected 'better text', got '{chunk0.text}'"
    assert chunk0.meta["ocr_engine"] == "trocr"
    assert chunk0.meta["ocr_confidence"] == pytest.approx(0.87)

    # Chunk 1 should remain marker
    chunk1 = all_chunks[1]
    assert chunk1.meta["ocr_engine"] == "marker"


@pytest.mark.asyncio
async def test_trocr_fires_on_handwriting_flag(fake_sf: FakeSessionFactory) -> None:
    hw_block = make_block("handwritten scrawl", confidence=0.75, is_handwriting=True)
    blocks = [hw_block]
    marker_out = make_marker_output(blocks)
    doc_meta = make_meta()

    with (
        patch("app.ingestion.orchestrator.run_marker", new_callable=AsyncMock, return_value=marker_out),
        patch(
            "app.ingestion.orchestrator.re_ocr_blocks",
            new_callable=AsyncMock,
            return_value=[("handwritten scrawl improved", 0.80)],
        ) as mock_trocr,
        patch("app.ingestion.orchestrator.extract_document_meta", new_callable=AsyncMock, return_value=(doc_meta, "ok")),
        patch("app.ingestion.orchestrator.deskew_pdf_bytes", side_effect=lambda b: b),
    ):
        from app.ingestion.orchestrator import ingest_document

        await ingest_document(
            file_bytes=b"%PDF-fake",
            filename="hw.pdf",
            mime="application/pdf",
            case_id=uuid.UUID(int=0),
            session_factory=fake_sf,
        )

    mock_trocr.assert_called_once()
