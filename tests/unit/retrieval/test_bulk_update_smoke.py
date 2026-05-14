"""Regression test: bulk executemany UPDATE with WHERE+bindparam succeeds.

Uses an in-memory SQLite engine (no pgvector dependency) to verify the exact
statement shape used in indexing.py doesn't raise the SQLAlchemy synchronize_session
error that would occur without .execution_options(synchronize_session=False).
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import Column, MetaData, String, Table, bindparam, update
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_bulk_update_executemany_with_synchronize_session_false() -> None:
    """The indexing.py executemany UPDATE pattern succeeds against a real engine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    meta = MetaData()
    chunks = Table(
        "chunks",
        meta,
        Column("id", String, primary_key=True),
        Column("embedding", String),  # JSON-encoded vec; avoids pgvector in SQLite
    )

    vec = json.dumps([0.1, 0.2, 0.3])
    ids = [str(uuid.uuid4()) for _ in range(5)]

    async with engine.begin() as conn:
        await conn.run_sync(meta.create_all)
        await conn.execute(chunks.insert(), [{"id": i, "embedding": None} for i in ids])

        stmt = (
            update(chunks)
            .where(chunks.c.id == bindparam("_id"))
            .values(embedding=bindparam("_embedding"))
            .execution_options(synchronize_session=False)
        )
        params = [{"_id": i, "_embedding": vec} for i in ids]
        await conn.execute(stmt, params)

        result = await conn.execute(chunks.select())
        rows = result.all()

    assert len(rows) == 5
    assert all(r.embedding == vec for r in rows)

    await engine.dispose()
