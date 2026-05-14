"""Ingestion orchestrator — the single entry point for processing any PDF or image file.

Handles: idempotency (sha256), image wrapping, deskew, Marker OCR, TrOCR fallback
(batched + concurrent with LLM extraction), structured extraction, chunk persistence
(embedding=NULL for phase 3). Never raises to the caller — errors captured in status.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import tempfile
import uuid
from pathlib import Path

import torch
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.db import SessionLocal
from app.ingestion.deskew import deskew_pdf_bytes
from app.ingestion.image_to_pdf import wrap_image_as_pdf
from app.ingestion.marker_runner import MarkerOutput, run_marker
from app.ingestion.models import Block
from app.ingestion.structured_extract import extract_document_meta
from app.ingestion.trocr_fallback import re_ocr_blocks
from app.models.pydantic_models import ChunkProvenance
from app.models.sqlalchemy_models import Chunk, Document, Page


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    mime: str,
    case_id: uuid.UUID,
    session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
) -> uuid.UUID:
    """Ingest a PDF or image file; return the document UUID.

    Idempotent: re-uploading the same bytes for the same case returns the existing id.
    All errors are recorded on the document row; this function never raises.
    """
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    doc_id = uuid.uuid4()

    async with session_factory() as session:
        # -------------------------------------------------------------------
        # 1. Idempotency check
        # -------------------------------------------------------------------
        existing = await session.execute(
            select(Document).where(
                Document.sha256 == sha256,
                Document.case_id == case_id,
            )
        )
        row = existing.scalars().first()
        if row is not None:
            logger.bind(doc_id=row.id, sha256=sha256).info("ingest_skipped_duplicate")
            return row.id  # type: ignore[return-value]

        # -------------------------------------------------------------------
        # 2. Image wrap
        # -------------------------------------------------------------------
        original_mime = mime
        if mime.startswith("image/"):
            file_bytes = wrap_image_as_pdf(file_bytes, mime)
            mime = "application/pdf"

        # -------------------------------------------------------------------
        # 3. Persist document row with status=ingesting
        # -------------------------------------------------------------------
        now = datetime.datetime.now(datetime.UTC)
        doc = Document(
            id=doc_id,
            case_id=case_id,
            original_filename=filename,
            mime=mime,
            sha256=sha256,
            status="ingesting",
            uploaded_at=now,
            meta={"original_mime": original_mime} if original_mime != mime else {},
        )
        session.add(doc)
        await session.commit()
        logger.bind(doc_id=doc_id, filename=filename, mime=mime).info("ingest_started")

        try:
            # ---------------------------------------------------------------
            # 4. Deskew (scanned PDFs only) + Marker OCR
            # ---------------------------------------------------------------
            pdf_bytes = await asyncio.to_thread(deskew_pdf_bytes, file_bytes)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(pdf_bytes)
                tmp.flush()
                marker_output: MarkerOutput = await run_marker(Path(tmp.name))

            blocks: list[Block] = marker_output.blocks
            low_conf_indices = [
                i
                for i, b in enumerate(blocks)
                if b.is_handwriting or b.confidence < settings.trocr_printed_threshold
            ]
            logger.bind(
                doc_id=doc_id,
                block_count=len(blocks),
                trocr_candidates=len(low_conf_indices),
            ).info("marker_done")

            # ---------------------------------------------------------------
            # 5. Collect crops for batch TrOCR
            # ---------------------------------------------------------------
            indexed_crops: list[tuple[int, object]] = []
            for idx in low_conf_indices:
                img = marker_output.get_block_image(idx)
                if img is not None:
                    indexed_crops.append((idx, img))

            # Release Marker's internal document/page objects; all crops are
            # already extracted as PIL images. Clearing the MPS cache here
            # frees Surya's cached tensors before TrOCR allocates its own.
            del marker_output
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # ---------------------------------------------------------------
            # 6. Run TrOCR batch + LLM extraction concurrently
            #    (LLM uses Marker-only text; TrOCR refines low-conf blocks
            #    for chunk storage but doesn't block extraction)
            # ---------------------------------------------------------------
            full_text = "\n".join(b.text for b in blocks)

            if indexed_crops:
                trocr_results, (meta, extract_status) = await asyncio.gather(
                    re_ocr_blocks([img for _, img in indexed_crops]),
                    extract_document_meta(full_text, doc_id),
                )
            else:
                trocr_results = []
                meta, extract_status = await extract_document_meta(full_text, doc_id)

            # Apply TrOCR refinements
            trocr_count = 0
            for (idx, _), (text, conf) in zip(indexed_crops, trocr_results, strict=False):
                block = blocks[idx]
                if conf > block.confidence:
                    blocks[idx] = block.model_copy(
                        update={"text": text, "confidence": conf, "ocr_engine": "trocr"}
                    )
                    trocr_count += 1

            logger.bind(doc_id=doc_id, block_count=trocr_count).info("trocr_fallback_done")

            # Free TrOCR's cached tensors before next document starts.
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # ---------------------------------------------------------------
            # 7. Persist pages
            # ---------------------------------------------------------------
            pages_by_num: dict[int, list[Block]] = {}
            for b in blocks:
                pages_by_num.setdefault(b.page, []).append(b)

            for page_num, page_blocks in pages_by_num.items():
                confs = [b.confidence for b in page_blocks]
                page_row = Page(
                    doc_id=doc_id,
                    page_number=page_num,
                    ocr_confidence_mean=sum(confs) / len(confs) if confs else None,
                    has_handwriting=any(b.is_handwriting for b in page_blocks),
                )
                session.add(page_row)

            # ---------------------------------------------------------------
            # 8. Persist chunks (embedding=NULL; phase 3 populates)
            # ---------------------------------------------------------------
            for block in blocks:
                provenance = ChunkProvenance(
                    bbox=block.bbox,
                    ocr_confidence=block.confidence,
                    ocr_engine=block.ocr_engine,
                    is_handwriting=block.is_handwriting,
                    low_ocr_confidence=block.confidence < 0.5,
                )
                chunk = Chunk(
                    doc_id=doc_id,
                    page_number=block.page,
                    section_heading=None,
                    char_offset_start=block.char_offset_start,
                    char_offset_end=block.char_offset_end,
                    text=block.text,
                    embedding=None,
                    parent_section_id=None,
                    meta=provenance.model_dump(),
                )
                session.add(chunk)

            chunk_count = len(blocks)
            logger.bind(doc_id=doc_id, chunk_count=chunk_count).info("persisted")

            # ---------------------------------------------------------------
            # 9. Finalise document
            # ---------------------------------------------------------------
            any_trocr = any(b.ocr_engine == "trocr" for b in blocks)
            total_pages = max(pages_by_num.keys(), default=0)
            final_status = "indexed" if extract_status == "ok" else "extraction_unclear"
            doc_meta = meta.model_dump(mode="json")
            if original_mime != mime:
                doc_meta["original_mime"] = original_mime

            doc.status = final_status
            doc.doc_type = meta.doc_type
            doc.total_pages = total_pages
            doc.ocr_engine = "marker+trocr" if any_trocr else "marker"
            doc.meta = doc_meta
            await session.commit()

        except Exception as exc:
            logger.bind(doc_id=doc_id, error=str(exc)).exception("ingest_failed")
            async with session_factory() as err_session:
                err_doc = await err_session.get(Document, doc_id)
                if err_doc is not None:
                    err_doc.status = "ingest_failed"
                    err_doc.meta = {"error": str(exc)}
                    await err_session.commit()

    return doc_id
