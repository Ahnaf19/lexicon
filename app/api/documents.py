"""Document upload, status, and metadata endpoints."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.ingestion.orchestrator import ingest_document
from app.models.sqlalchemy_models import Chunk, Document, Page

router = APIRouter()

_ALLOWED_MIMES = {"application/pdf", "image/jpeg", "image/png"}


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    status: str


class DocumentStatusResponse(BaseModel):
    status: str
    last_event_at: str


class DocumentDetailResponse(BaseModel):
    document_id: uuid.UUID
    original_filename: str
    status: str
    doc_type: str | None
    ocr_engine: str | None
    total_pages: int | None
    page_count: int
    chunk_count: int
    meta: dict[str, Any]


@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    file: UploadFile,
    case_id: uuid.UUID = uuid.UUID(int=0),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_session),
) -> UploadResponse:
    mime = file.content_type or ""
    if mime not in _ALLOWED_MIMES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{mime}'. Accepted: {sorted(_ALLOWED_MIMES)}",
        )

    file_bytes = await file.read()
    filename = file.filename or "upload"
    doc_id = uuid.uuid4()

    background_tasks.add_task(
        ingest_document,
        file_bytes=file_bytes,
        filename=filename,
        mime=mime,
        case_id=case_id,
    )
    return UploadResponse(document_id=doc_id, status="queued")


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> DocumentStatusResponse:
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(
        status=doc.status,
        last_event_at=doc.uploaded_at.isoformat(),
    )


@router.get("/{doc_id}", response_model=DocumentDetailResponse)
async def get_document(
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> DocumentDetailResponse:
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    page_count_row = await session.execute(
        select(func.count(Page.id)).where(Page.doc_id == doc_id)
    )
    page_count: int = page_count_row.scalar_one()

    chunk_count_row = await session.execute(
        select(func.count(Chunk.id)).where(Chunk.doc_id == doc_id)
    )
    chunk_count: int = chunk_count_row.scalar_one()

    return DocumentDetailResponse(
        document_id=doc.id,
        original_filename=doc.original_filename,
        status=doc.status,
        doc_type=doc.doc_type,
        ocr_engine=doc.ocr_engine,
        total_pages=doc.total_pages,
        page_count=page_count,
        chunk_count=chunk_count,
        meta=doc.meta or {},
    )
