"""Happy path: 3 high-confidence blocks → 1 document, 3 chunks, status=ocr_done."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import FakeSessionFactory, make_block, make_marker_output, make_meta


@pytest.mark.asyncio
async def test_happy_path(fake_sf: FakeSessionFactory) -> None:
    blocks = [
        make_block("Clause one.", confidence=0.92),
        make_block("Clause two.", confidence=0.88, page=1),
        make_block("Clause three.", confidence=0.95, page=2),
    ]
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
        ) as mock_trocr,
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

        doc_id = await ingest_document(
            file_bytes=b"%PDF-fake",
            filename="test.pdf",
            mime="application/pdf",
            case_id=uuid.UUID(int=0),
            session_factory=fake_sf,
        )

    # TrOCR must NOT have been called — all blocks are above the 0.4 threshold
    mock_trocr.assert_not_called()

    # There should be at least one session used
    assert len(fake_sf.sessions) >= 1

    # Collect all added objects across sessions
    all_added: dict[str, list] = {}
    for s in fake_sf.sessions:
        for cls_name, rows in s.added.items():
            all_added.setdefault(cls_name, []).extend(rows)

    assert len(all_added.get("Document", [])) == 1, "Expected exactly 1 Document row"
    assert len(all_added.get("Chunk", [])) == 3, "Expected 3 Chunk rows"

    doc_row = all_added["Document"][0]
    assert doc_row.status == "ocr_done"
    assert doc_row.id == doc_id
