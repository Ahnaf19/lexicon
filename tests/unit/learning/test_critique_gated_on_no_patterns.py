"""L3 gate: critique returns {} with zero LLM calls when no patterns apply."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.generation.nodes.critique import critique
from app.models.pydantic_models import ChecklistItem


def _make_item() -> ChecklistItem:
    return ChecklistItem(
        id=uuid.uuid4(),
        category="Parties",
        title="Buyer identified",
        description="Verify buyer",
        status="unclear",
        required=True,
        evidence=[],
        confidence=0.5,
        rationale="Insufficient evidence",
    )


@pytest.mark.asyncio
async def test_no_patterns_returns_empty_no_llm_call() -> None:
    """When learned_patterns is empty, critique must return {} without calling the LLM."""
    item = _make_item()
    state = {
        "current_item_slug": "buyer-identified",
        "items_in_progress": {"buyer-identified": item},
        "learned_patterns": [],  # empty — L3 gate should fire
    }

    call_count = 0

    def _counting_get_chat_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return MagicMock()

    with patch("app.generation.nodes.critique.get_chat_model", _counting_get_chat_model):
        result = await critique(state)

    assert result == {}, f"Expected empty dict, got {result}"
    assert call_count == 0, f"LLM was called {call_count} time(s); expected 0 (L3 gate)"


@pytest.mark.asyncio
async def test_empty_state_returns_empty() -> None:
    """No current_item_slug → return {} safely."""
    result = await critique({"items_in_progress": {}, "learned_patterns": []})
    assert result == {}
