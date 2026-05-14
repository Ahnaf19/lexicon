"""Structured-output happy path for _call_extraction_llm.

Verifies that the function calls with_structured_output(DraftPatternBatch) — not
the broken list[DraftLearnedPattern] generic alias — and unwraps batch.patterns.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.learning.pattern_extractor import _call_extraction_llm
from app.models.pydantic_models import DraftLearnedPattern, DraftPatternBatch


@pytest.mark.asyncio
async def test_call_extraction_llm_uses_batch_schema_and_unwraps_patterns() -> None:
    fake_batch = DraftPatternBatch(
        patterns=[
            DraftLearnedPattern(
                pattern_type="rename_rule",
                doc_type_scope="commercial_contract",
                rule_json={"from_text": "Counterparty", "to_text": "Borrower"},
                supporting_edit_ids=[uuid.uuid4()],
                rationale="Repeated rename observed.",
            )
        ]
    )

    structured_model = MagicMock()
    structured_model.ainvoke = AsyncMock(return_value=fake_batch)

    chat_model = MagicMock()
    chat_model.with_structured_output = MagicMock(return_value=structured_model)

    with patch(
        "app.learning.pattern_extractor.get_chat_model", return_value=chat_model
    ):
        result = await _call_extraction_llm("[]", "[]")

    # Schema must be the concrete wrapper model, not the generic alias.
    chat_model.with_structured_output.assert_called_once_with(DraftPatternBatch)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].pattern_type == "rename_rule"


@pytest.mark.asyncio
async def test_call_extraction_llm_returns_empty_list_when_no_patterns() -> None:
    fake_batch = DraftPatternBatch(patterns=[])

    structured_model = MagicMock()
    structured_model.ainvoke = AsyncMock(return_value=fake_batch)

    chat_model = MagicMock()
    chat_model.with_structured_output = MagicMock(return_value=structured_model)

    with patch(
        "app.learning.pattern_extractor.get_chat_model", return_value=chat_model
    ):
        result = await _call_extraction_llm("[]", "[]")

    assert result == []
