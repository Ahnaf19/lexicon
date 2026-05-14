"""P1 invariant: patterns with no supporting_edit_ids are rejected by the extractor."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.learning.pattern_extractor import _extract_patterns_inner


@pytest.fixture
def fake_checklist_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def _mock_session_factory(fake_checklist_id: uuid.UUID):
    """Minimal mocks so _extract_patterns_inner can run without a real DB."""
    from unittest.mock import MagicMock

    checklist_mock = MagicMock()
    checklist_mock.template_id = uuid.uuid4()
    checklist_mock.eval_metrics = {}

    template_mock = MagicMock()
    template_mock.doc_type = "commercial_contract"

    # Patch the heavy I/O surfaces.
    with (
        patch("app.learning.pattern_extractor.SessionLocal") as mock_sl,
        patch("app.learning.pattern_extractor._call_extraction_llm") as mock_llm,
        patch("app.learning.pattern_extractor._write_few_shot_examples", new_callable=AsyncMock),
    ):
        # Session context manager.
        session_mock = AsyncMock()
        session_mock.get = AsyncMock(side_effect=[checklist_mock, template_mock])

        # edit_events query returns one event.
        event_row = MagicMock()
        event_row.id = uuid.uuid4()
        event_row.checklist_id = fake_checklist_id
        event_row.checklist_item_id = None
        event_row.event_type = "item_renamed"
        event_row.payload = {"event_type": "item_renamed", "old_title": "A", "new_title": "B"}
        event_row.created_at = datetime.now(timezone.utc)

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [event_row]
        execute_result = MagicMock()
        execute_result.scalars.return_value = scalars_mock
        session_mock.execute = AsyncMock(return_value=execute_result)

        session_mock.add = MagicMock()
        session_mock.commit = AsyncMock()

        context_mock = AsyncMock()
        context_mock.__aenter__ = AsyncMock(return_value=session_mock)
        context_mock.__aexit__ = AsyncMock(return_value=None)
        mock_sl.return_value = context_mock

        # LLM returns a DraftLearnedPattern with EMPTY supporting_edit_ids → P1 must reject.
        from app.models.pydantic_models import DraftLearnedPattern

        mock_llm.return_value = [
            DraftLearnedPattern(
                pattern_type="rename_rule",
                doc_type_scope="commercial_contract",
                rule_json={"from_text": "Counterparty", "to_text": "Borrower"},
                supporting_edit_ids=[],  # deliberately empty — P1 trigger
                rationale="Invented without evidence.",
            )
        ]

        yield fake_checklist_id, session_mock, mock_sl


@pytest.mark.asyncio
async def test_p1_rejects_empty_supporting_ids(_mock_session_factory) -> None:
    checklist_id, session_mock, mock_sl = _mock_session_factory
    # Run the extractor — it should log a warning and NOT call session.add for LearnedPattern.
    await _extract_patterns_inner(checklist_id)
    # session.add must never have been called with a LearnedPatternORM (pattern was rejected).
    from app.models.sqlalchemy_models import LearnedPattern as LearnedPatternORM

    for call in session_mock.add.call_args_list:
        obj = call[0][0]
        assert not isinstance(obj, LearnedPatternORM), (
            "P1 violated: extractor persisted a pattern with no supporting edit ids"
        )
