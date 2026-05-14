"""Document indexing — orchestrates rechunking + embedding + status promotion (PRD §5c).

Flow:
  1. rechunk_document()  — replace OCR blocks with section/window chunks
  2. embed_texts()       — window chunks only (section rows stay embedding=NULL)
  3. bulk UPDATE         — write embeddings back
  4. status → "indexed"  — document is now retrieval-ready
"""

from __future__ import annotations

import time
import uuid

from loguru import logger
from sqlalchemy import bindparam, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.db import SessionLocal
from app.models.sqlalchemy_models import Chunk, Document
from app.retrieval.chunking import rechunk_document
from app.retrieval.embedding import _stable_batch_size, embed_texts, warmup

# Module-level warmup guard (P10)
_warmed: bool = False


async def index_document(
    doc_id: uuid.UUID,
    session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
) -> None:
    """Rechunk, embed, and mark doc retrieval-ready.

    Idempotent: re-running replaces all chunks and re-embeds from scratch.
    """
    global _warmed

    if not _warmed:
        await warmup()
        _warmed = True

    async with session_factory() as session:
        t0 = time.monotonic()

        # Fast-path: if windows already exist with NULL embedding, skip rechunking.
        # This avoids DELETE+re-INSERT churn when retrying after a failed embed step.
        pending_result = await session.execute(
            select(func.count())
            .select_from(Chunk)
            .where(
                Chunk.doc_id == doc_id,
                Chunk.meta["kind"].as_string() == "window",
                Chunk.embedding.is_(None),
            )
        )
        pending_windows = pending_result.scalar_one()

        if pending_windows > 0:
            window_count = pending_windows
            section_count = 0
            logger.bind(doc_id=doc_id, pending_windows=pending_windows).info(
                "index_fast_path_embed_only"
            )
        else:
            # 1. Rechunk
            window_count, section_count = await rechunk_document(doc_id, session)
            rechunk_elapsed = round(time.monotonic() - t0, 2)
            logger.bind(
                doc_id=doc_id,
                window_count=window_count,
                section_count=section_count,
                elapsed_sec=rechunk_elapsed,
            ).info("rechunk_done")


        if window_count == 0:
            logger.bind(doc_id=doc_id).warning("index_no_windows")
            return

        # 2. Fetch window chunk ids + texts
        result = await session.execute(
            select(Chunk.id, Chunk.text)
            .where(
                Chunk.doc_id == doc_id,
                Chunk.meta["kind"].as_string() == "window",
            )
            .order_by(Chunk.id)
        )
        rows = result.all()
        chunk_ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]

        # 3. Embed (window chunks only — P1)
        t1 = time.monotonic()
        vectors = await embed_texts(texts)
        embed_elapsed = round(time.monotonic() - t1, 2)

        # 4. Bulk UPDATE embeddings (P4) — Core-level executemany bypasses ORM.
        # update(Chunk.__table__) instead of update(Chunk) avoids the ORM "Bulk UPDATE
        # by Primary Key" path which requires params keyed by the PK column name and
        # cannot accept WHERE-clause bindparams as aliases.
        _tbl = Chunk.__table__
        stmt = (
            update(_tbl)
            .where(_tbl.c.id == bindparam("_id"))
            .values(embedding=bindparam("_embedding"))
        )
        params = [
            {"_id": cid, "_embedding": vec}
            for cid, vec in zip(chunk_ids, vectors)
        ]
        await session.execute(stmt, params)

        # 5. Promote document status to "indexed"
        doc = await session.get(Document, doc_id)
        if doc is not None and doc.status == "ocr_done":
            doc.status = "indexed"

        await session.commit()

        logger.bind(
            doc_id=doc_id,
            count=len(chunk_ids),
            elapsed_sec=embed_elapsed,
            final_batch_size=_stable_batch_size,
        ).info("embedding_done")
