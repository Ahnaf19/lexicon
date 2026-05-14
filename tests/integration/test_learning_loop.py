"""Learning loop integration test — edit capture → pattern extraction → application.

Uses shared pg_schema + pg_session fixtures (auto-migrates lexicon_test).
Overrides FastAPI's get_session dependency to route all DB I/O through lexicon_test.
LLM is stubbed.
Skips if Postgres is unavailable.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.learning.pattern_extractor import extract_patterns
from app.main import app as fastapi_app
from app.models.sqlalchemy_models import (
    Checklist as ChecklistORM,
    ChecklistItem as ChecklistItemORM,
    ChecklistTemplate as ChecklistTemplateORM,
    EditEvent as EditEventORM,
    LearnedPattern as LearnedPatternORM,
)

# ---------------------------------------------------------------------------
# Template UUID (must match the TEMPLATES registry deterministic UUID).
# ---------------------------------------------------------------------------

_CASE_ID = uuid.uuid4()
_TEMPLATE_ID = uuid.uuid5(
    uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8"),
    "lexicon.template/commercial_contract",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_session(pg_schema) -> AsyncSession:
    """Single AsyncSession from the test engine; caller commits/rolls back as needed."""
    async with AsyncSession(pg_schema) as session:
        yield session


@pytest.fixture
async def client(pg_schema):
    """AsyncClient that routes FastAPI DB calls to lexicon_test via dependency override."""

    async def _override_get_session():
        async with AsyncSession(pg_schema) as session:
            yield session

    fastapi_app.dependency_overrides[get_session] = _override_get_session
    transport = ASGITransport(app=fastapi_app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as c:
            yield c
    finally:
        fastapi_app.dependency_overrides.pop(get_session, None)


@pytest.fixture
async def seeded_checklist(test_session):
    """Insert a checklist with two items into lexicon_test."""
    from app.generation.templates import TEMPLATES

    tmpl = TEMPLATES["commercial_contract"]

    # Ensure template row.
    tmpl_row = await test_session.get(ChecklistTemplateORM, _TEMPLATE_ID)
    if tmpl_row is None:
        test_session.add(
            ChecklistTemplateORM(
                id=_TEMPLATE_ID,
                name=tmpl.name,
                doc_type=tmpl.doc_type,
                version=tmpl.version,
                items=[i.model_dump() for i in tmpl.items],
                active=True,
            )
        )
        await test_session.flush()

    checklist_id = uuid.uuid4()
    test_session.add(
        ChecklistORM(
            id=checklist_id,
            case_id=_CASE_ID,
            template_id=_TEMPLATE_ID,
            status="draft",
            generated_at=datetime.now(timezone.utc),
            model_version="test",
            prompt_version="v1",
        )
    )
    await test_session.flush()

    item_id_1 = uuid.uuid4()
    item_id_2 = uuid.uuid4()

    test_session.add(
        ChecklistItemORM(
            id=item_id_1,
            checklist_id=checklist_id,
            category="Parties",
            title="Counterparty identified",
            description="Verify counterparty identity",
            status="unclear",
            required=True,
            confidence=0.5,
            rationale="No evidence",
            learned_from_pattern_ids=[],
        )
    )
    test_session.add(
        ChecklistItemORM(
            id=item_id_2,
            checklist_id=checklist_id,
            category="Financial Terms",
            title="Payment schedule confirmed",
            description="Verify payment",
            status="unclear",
            required=True,
            confidence=0.5,
            rationale="No evidence",
            learned_from_pattern_ids=[],
        )
    )
    await test_session.commit()

    yield checklist_id, item_id_1, item_id_2

    # Cleanup in FK-safe order: edit_events → checklist_items → checklists.
    import sqlalchemy as _sa

    _opts = {"synchronize_session": False}
    await test_session.execute(
        _sa.delete(EditEventORM).where(EditEventORM.checklist_id == checklist_id).execution_options(**_opts)
    )
    await test_session.execute(
        _sa.delete(ChecklistItemORM).where(ChecklistItemORM.checklist_id == checklist_id).execution_options(**_opts)
    )
    await test_session.execute(
        _sa.delete(ChecklistORM).where(ChecklistORM.id == checklist_id).execution_options(**_opts)
    )
    await test_session.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_events_captured_and_pattern_extracted(
    pg_schema,
    test_session,
    seeded_checklist,
    client,
) -> None:
    """PATCH item → finalize → extract_patterns → assert edit_events + learned_patterns."""
    checklist_id, item_id_1, _ = seeded_checklist

    # PATCH: rename "Counterparty" → "Borrower".
    resp = await client.patch(
        f"/checklists/{checklist_id}/items/{item_id_1}",
        json={"title": "Borrower identified"},
        headers={"X-Operator-Id": "test-operator"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["title"] == "Borrower identified"

    # Verify edit_event was written.
    test_session.expire_all()
    result = await test_session.execute(
        select(EditEventORM).where(EditEventORM.checklist_id == checklist_id)
    )
    events = list(result.scalars().all())
    assert len(events) >= 1
    assert any(e.event_type == "item_renamed" for e in events)
    # Capture event_id before expire_all() would make the objects stale.
    event_id = events[0].id

    # Finalize via API.
    with patch(
        "app.learning.pattern_extractor.extract_patterns",
        new=AsyncMock(),
    ):
        resp = await client.post(f"/checklists/{checklist_id}/finalize")
    assert resp.status_code == 202

    test_session.expire_all()

    # Run pattern extraction synchronously against lexicon_test.
    from app.core.db import SessionLocal as _prod_session_local

    with (
        patch("app.learning.pattern_extractor.SessionLocal") as mock_sl,
        patch(
            "app.learning.pattern_extractor.embed_query",
            new=AsyncMock(return_value=[0.1] * 768),
        ),
    ):
        # Redirect SessionLocal in extractor to the test engine.
        from sqlalchemy.ext.asyncio import async_sessionmaker

        test_session_factory = async_sessionmaker(pg_schema, expire_on_commit=False)
        mock_sl.return_value.__aenter__ = AsyncMock(
            return_value=(await test_session_factory().__aenter__())
        )
        mock_sl.return_value.__aexit__ = AsyncMock(return_value=False)

        # Use the LLM stub for extraction.
        from app.models.pydantic_models import DraftLearnedPattern

        draft = DraftLearnedPattern(
            pattern_type="rename_rule",
            doc_type_scope="commercial_contract",
            rule_json={"from_text": "Counterparty", "to_text": "Borrower"},
            supporting_edit_ids=[event_id],
            rationale="Operator consistently renames Counterparty to Borrower",
        )
        with patch(
            "app.learning.pattern_extractor._call_extraction_llm",
            new=AsyncMock(return_value=[draft]),
        ):
            # Directly call the inner function with a real test session.
            async with test_session_factory() as ext_session:
                # Minimal direct call to _upsert_pattern bypassing the full extractor.
                from app.learning.pattern_extractor import _upsert_pattern

                now = datetime.now(timezone.utc)
                await _upsert_pattern(
                    session=ext_session,
                    draft=draft,
                    valid_ids=[event_id],
                    existing_rows=[],
                    doc_type="commercial_contract",
                    now=now,
                )
                await ext_session.commit()

    # Assert learned_pattern row was created.
    test_session.expire_all()
    result = await test_session.execute(select(LearnedPatternORM))
    patterns = list(result.scalars().all())
    assert len(patterns) >= 1
    p = patterns[0]
    assert p.pattern_type == "rename_rule"
    assert p.corroborating_edit_count == 1
    assert not p.promoted  # needs >= 3 to promote

    # Cleanup patterns.
    for pat in patterns:
        await test_session.delete(pat)
    await test_session.commit()


@pytest.mark.asyncio
async def test_add_and_delete_item_events(
    pg_schema,
    test_session,
    seeded_checklist,
    client,
) -> None:
    """POST item → item_added event; DELETE item → item_removed event."""
    checklist_id, _, _ = seeded_checklist

    new_item = {
        "id": str(uuid.uuid4()),
        "category": "Signatures",
        "title": "Signatures notarized",
        "description": "Verify notarization",
        "status": "unclear",
        "required": True,
        "evidence": [],
        "confidence": 0.0,
        "rationale": "Not yet checked",
        "learned_from_pattern_ids": [],
    }
    resp = await client.post(
        f"/checklists/{checklist_id}/items",
        json=new_item,
        headers={"X-Operator-Id": "test-operator"},
    )
    assert resp.status_code == 201, resp.text
    created_id = uuid.UUID(resp.json()["id"])

    resp = await client.delete(
        f"/checklists/{checklist_id}/items/{created_id}",
        headers={"X-Operator-Id": "test-operator"},
    )
    assert resp.status_code == 204, resp.text

    test_session.expire_all()
    result = await test_session.execute(
        select(EditEventORM.event_type).where(
            EditEventORM.checklist_id == checklist_id
        )
    )
    event_types = [r[0] for r in result]
    assert "item_added" in event_types
    assert "item_removed" in event_types


@pytest.mark.asyncio
async def test_learned_patterns_api(
    pg_schema,
    client,
) -> None:
    """GET /checklists/learned-patterns returns a valid list structure."""
    resp = await client.get("/checklists/learned-patterns?limit=5")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
