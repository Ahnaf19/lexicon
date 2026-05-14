"""Promoted template_addition pattern injects new item into template.items."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.generation.templates.types import TemplateItem
from app.models.pydantic_models import LearnedPattern


def _make_promoted_addition_pattern(new_item: TemplateItem) -> LearnedPattern:
    return LearnedPattern(
        id=uuid.uuid4(),
        pattern_type="template_addition",
        doc_type_scope="commercial_contract",
        rule_json={"item_template": new_item.model_dump()},
        supporting_edit_ids=[uuid.uuid4(), uuid.uuid4(), uuid.uuid4()],
        confidence=0.95,
        corroborating_edit_count=3,
        promoted=True,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_template_addition_injects_item() -> None:
    """A promoted template_addition pattern should append an item to the active template."""
    from app.generation.nodes.load_template import load_template
    from app.generation.templates import TEMPLATES

    extra_item = TemplateItem(
        slug="signatures-notarized",
        title="Signatures dated and notarized",
        description="All signatures include date and notarization",
        category="Signatures",
        required=True,
        sub_query="notarized signature",
    )
    pattern = _make_promoted_addition_pattern(extra_item)

    state = {"template_slug": "commercial_contract"}

    original_template = TEMPLATES["commercial_contract"]
    original_slugs = {i.slug for i in original_template.items}

    # Mock the DB calls: upsert + pattern query.
    mock_session = AsyncMock()

    scalars_mock = MagicMock()
    orm_row = MagicMock()
    orm_row.id = pattern.id
    orm_row.pattern_type = pattern.pattern_type
    orm_row.doc_type_scope = pattern.doc_type_scope
    orm_row.rule_json = pattern.rule_json
    orm_row.supporting_edit_ids = [str(e) for e in pattern.supporting_edit_ids]
    orm_row.confidence = pattern.confidence
    orm_row.corroborating_edit_count = pattern.corroborating_edit_count
    orm_row.promoted = pattern.promoted
    orm_row.created_at = pattern.created_at

    scalars_mock.all.return_value = [orm_row]
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars_mock
    mock_session.execute = AsyncMock(return_value=execute_result)
    mock_session.commit = AsyncMock()

    context = AsyncMock()
    context.__aenter__ = AsyncMock(return_value=mock_session)
    context.__aexit__ = AsyncMock(return_value=None)

    with patch("app.generation.nodes.load_template.SessionLocal", return_value=context):
        result = await load_template(state)

    active_template = result["template"]
    active_slugs = {i.slug for i in active_template.items}

    assert "signatures-notarized" in active_slugs, (
        "Promoted template_addition pattern did not inject the new item"
    )
    assert original_slugs.issubset(active_slugs), "Original items must be preserved"
    learned: list = result["learned_patterns"]  # type: ignore[assignment]
    assert len(learned) == 1
