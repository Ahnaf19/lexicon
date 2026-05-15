"""FastAPI application entry point."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.checklists import router as checklists_router
from app.api.documents import router as documents_router
from app.core.db import SessionLocal, get_session
from app.core.logging import configure_logging
from app.models.pydantic_models import EvidenceCitation
from app.models.sqlalchemy_models import EvidenceCitation as EvidenceCitationORM


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    yield


app = FastAPI(title="Lexicon", version="0.1.0", lifespan=lifespan)
app.include_router(documents_router, prefix="/documents", tags=["documents"])
app.include_router(checklists_router, prefix="/checklists", tags=["checklists"])


@app.get("/healthz", tags=["meta"])
async def healthz() -> JSONResponse:
    """DB + basic reachability check (PRD §8)."""
    checks: dict[str, str] = {}
    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    ok = all(v == "ok" for v in checks.values())
    return JSONResponse(content={"status": "ok" if ok else "degraded", **checks}, status_code=200)


@app.get("/evidence/{citation_id}", response_model=EvidenceCitation, tags=["meta"])
async def get_evidence(
    citation_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> EvidenceCitation:
    """Return a single evidence citation by ID — chunk + snippet + provenance (PRD §8)."""
    row = await session.get(EvidenceCitationORM, citation_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Citation {citation_id} not found")
    return EvidenceCitation(
        citation_id=row.id,
        chunk_id=row.chunk_id,
        doc_id=row.doc_id,
        page_number=row.page_number,
        char_offset_start=row.char_offset_start,
        char_offset_end=row.char_offset_end,
        snippet=row.snippet,
        retrieval_score=row.retrieval_score,
        rerank_score=row.rerank_score,
    )
