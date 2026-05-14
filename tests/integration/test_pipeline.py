"""End-to-end pipeline integration test with real Postgres and stubbed LLM.

Skips automatically if Postgres is unreachable (unit tests still pass).

LLM is stubbed by patching `app.generation.nodes.draft_item._call_llm` directly,
bypassing the `with_structured_output` call which FakeListChatModel does not support.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.generation.graph import build_graph
from app.generation.state import ChecklistState, DraftChecklistItem
from app.generation.templates import TEMPLATES
from app.models.pydantic_models import Checklist
from app.models.sqlalchemy_models import (
    Checklist as ChecklistORM,
    Document,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TEMPLATE = TEMPLATES["commercial_contract"]
_CASE_ID = uuid.uuid4()


@pytest.fixture(scope="module")
def engine():
    """Async engine for lexicon_test; skips if unreachable."""
    import sqlalchemy
    from sqlalchemy.engine.url import make_url
    from sqlalchemy.pool import NullPool

    # Use URL object so only the database name is swapped (str.replace would also
    # corrupt the username).  NullPool avoids event-loop-binding issues with asyncpg.
    test_url_obj = make_url(str(settings.db_url)).set(database="lexicon_test")
    eng = create_async_engine(test_url_obj, echo=False, poolclass=NullPool)

    async def _ping() -> None:
        async with eng.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))

    try:
        asyncio.run(_ping())
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")

    yield eng

    asyncio.run(eng.dispose())


@pytest.fixture(scope="module")
def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture(scope="module")
def seeded_case(session_factory):
    """Insert two indexed docs with window chunks; committed before the graph runs."""
    doc_id_1 = uuid.uuid4()
    doc_id_2 = uuid.uuid4()

    async def _seed() -> None:
        async with session_factory() as session:
            for doc_id, dtype in [(doc_id_1, "commercial_contract"), (doc_id_2, "commercial_contract")]:
                session.add(
                    Document(
                        id=doc_id,
                        case_id=_CASE_ID,
                        original_filename=f"test_{doc_id}.pdf",
                        sha256=f"sha256_{doc_id}",
                        doc_type=dtype,
                        uploaded_at=datetime.now(timezone.utc),
                        status="indexed",
                        meta={},
                    )
                )
            await session.flush()

            zero_vec = "[" + ",".join(["0.0"] * 768) + "]"
            for chunk_text, doc_id in [
                ("The parties to this agreement are Acme Corp and Zeta LLC.", doc_id_1),
                ("Governing law shall be the State of Delaware.", doc_id_2),
            ]:
                await session.execute(
                    text(
                        """
                        INSERT INTO chunks
                            (id, doc_id, page_number, char_offset_start, char_offset_end,
                             text, embedding, parent_section_id, meta)
                        VALUES
                            (:id, :doc_id, 1, 0, :end_off, :txt,
                             CAST(:vec AS vector), NULL, CAST(:meta AS jsonb))
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "doc_id": str(doc_id),
                        "end_off": len(chunk_text),
                        "txt": chunk_text,
                        "vec": zero_vec,
                        "meta": json.dumps({"kind": "window"}),
                    },
                )
            await session.commit()

    asyncio.run(_seed())
    return {"case_id": _CASE_ID, "doc_id_1": doc_id_1, "doc_id_2": doc_id_2}


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline(seeded_case, session_factory):
    """Full graph.ainvoke with stubbed _call_llm; verify persistence invariants."""
    case_id = seeded_case["case_id"]
    doc_id_1 = seeded_case["doc_id_1"]

    # Build canned DraftChecklistItem objects (12 items, alternating present/unclear).
    # We patch _call_llm directly so with_structured_output is never called.
    canned: list[DraftChecklistItem] = []
    for i in range(12):
        if i % 3 == 0:
            canned.append(DraftChecklistItem(
                status="present",
                confidence=0.85,
                rationale=f"[doc={doc_id_1} p.1] Evidence found for item {i}.",
                cited_evidence=["E1"],
            ))
        else:
            canned.append(DraftChecklistItem(
                status="unclear",
                confidence=0.3,
                rationale=f"Insufficient evidence for item {i}.",
                cited_evidence=[],
            ))

    call_count = 0

    async def _fake_call_llm(prompt_text: str, use_strict: bool = False) -> DraftChecklistItem:
        nonlocal call_count
        result = canned[min(call_count, len(canned) - 1)]
        call_count += 1
        return result

    noop_warmup = AsyncMock(return_value=None)

    with patch("app.generation.nodes.draft_item._call_llm", side_effect=_fake_call_llm), \
         patch("app.generation.warmup.warmup_llm", side_effect=noop_warmup), \
         patch("app.generation.nodes.classify_doc_set.SessionLocal", session_factory), \
         patch("app.generation.nodes.load_template.SessionLocal", session_factory), \
         patch("app.generation.nodes.retrieve_evidence.SessionLocal", session_factory), \
         patch("app.generation.nodes.assemble.SessionLocal", session_factory):

        graph = build_graph().compile()

        initial_state: ChecklistState = {
            "case_id": case_id,
            "template_slug": "commercial_contract",
            "document_ids": [],
            "errors": [],
        }

        final_state = await graph.ainvoke(initial_state)

    # --- Assertions ---
    checklist = final_state.get("checklist")
    assert checklist is not None, "assemble node must return a checklist in state"
    assert isinstance(checklist, Checklist)
    assert checklist.prompt_version == "v1"

    # DB persistence
    async with session_factory() as session:
        result = await session.execute(
            select(ChecklistORM).where(ChecklistORM.case_id == case_id)
        )
        cl_row = result.scalar_one_or_none()

    assert cl_row is not None, "Checklist ORM row must be persisted"
    assert cl_row.prompt_version == "v1"

    # CRITICAL grounding invariant: no present item may lack evidence citations.
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT ci.id, ci.title, ci.status
                FROM checklist_items ci
                WHERE ci.checklist_id = :cid
                  AND ci.status = 'present'
                  AND NOT EXISTS (
                      SELECT 1 FROM evidence_citations ec
                      WHERE ec.checklist_item_id = ci.id
                  )
                """
            ),
            {"cid": str(cl_row.id)},
        )
        violations = result.all()

    assert violations == [], (
        f"Grounding invariant violated — present items without evidence: "
        f"{[(str(r[0]), r[1]) for r in violations]}"
    )

    # LLM was called at least once per template item.
    assert call_count == len(_TEMPLATE.items), (
        f"Expected {len(_TEMPLATE.items)} LLM calls, got {call_count}"
    )
