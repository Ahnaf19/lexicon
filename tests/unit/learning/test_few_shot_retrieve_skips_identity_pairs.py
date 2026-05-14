"""P3: few-shot pairs where original_draft == final_item are excluded."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.generation.templates.types import TemplateItem
from app.learning.few_shot_bank import retrieve_few_shot
from app.models.pydantic_models import ChecklistItem

_ITEM_TMPL = TemplateItem(
    slug="buyer-identified",
    title="Buyer identified",
    description="Verify buyer identity",
    category="Parties",
    required=True,
    sub_query="buyer identification",
)

_CL_ITEM = ChecklistItem(
    id=uuid.uuid4(),
    category="Parties",
    title="Buyer identified",
    description="Verify",
    status="unclear",
    required=True,
    evidence=[],
    confidence=0.5,
    rationale="No evidence",
)


def _make_session(rows: list) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.__iter__ = MagicMock(return_value=iter(rows))
    session.execute = AsyncMock(return_value=result)
    return session


@pytest.mark.asyncio
async def test_identity_pair_excluded() -> None:
    """A pair where original_draft == final_item must be filtered out (P3)."""
    # Serialize as JSONB dicts (both identical).
    item_dict = _CL_ITEM.model_dump(mode="json")

    session = _make_session([(item_dict, item_dict)])  # one identity pair

    with patch("app.learning.few_shot_bank.embed_query", return_value=[0.1] * 768):
        pairs = await retrieve_few_shot(
            session=session,
            template_item=_ITEM_TMPL,
            doc_type="commercial_contract",
            evidence_summary="",
            k=3,
        )

    assert pairs == [], "Identity pair must be excluded from few-shot results"


@pytest.mark.asyncio
async def test_differing_pair_included() -> None:
    """A pair where original_draft differs from final_item is included."""
    orig_dict = _CL_ITEM.model_dump(mode="json")
    final_dict = _CL_ITEM.model_copy(update={"title": "Buyer verified and notarized"}).model_dump(
        mode="json"
    )

    session = _make_session([(orig_dict, final_dict)])

    with patch("app.learning.few_shot_bank.embed_query", return_value=[0.1] * 768):
        pairs = await retrieve_few_shot(
            session=session,
            template_item=_ITEM_TMPL,
            doc_type="commercial_contract",
            evidence_summary="",
            k=3,
        )

    assert len(pairs) == 1
    original, final = pairs[0]
    assert original.title != final.title
