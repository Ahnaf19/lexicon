"""Few-shot bank: examples from doc_type=A must not surface for doc_type=B queries."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.generation.templates.types import TemplateItem
from app.learning.few_shot_bank import retrieve_few_shot

_ITEM = TemplateItem(
    slug="buyer-identified",
    title="Buyer identified",
    description="Verify buyer identity",
    category="Parties",
    required=True,
    sub_query="buyer identification",
)


@pytest.mark.asyncio
async def test_doc_type_filter_respected() -> None:
    """The SQL query is parameterised with doc_type; wrong-type rows must not appear."""
    session_mock = AsyncMock()

    # Simulate empty result for doc_type=nda (no examples seeded for that type).
    execute_result = MagicMock()
    execute_result.__iter__ = MagicMock(return_value=iter([]))
    session_mock.execute = AsyncMock(return_value=execute_result)

    with patch("app.learning.few_shot_bank.embed_query", return_value=[0.1] * 768):
        pairs = await retrieve_few_shot(
            session=session_mock,
            template_item=_ITEM,
            doc_type="nda",
            evidence_summary="some evidence",
            k=3,
        )

    assert pairs == [], "Expected empty results for doc_type with no seeded examples"

    # Verify the query was called with doc_type="nda".
    call_kwargs = session_mock.execute.call_args
    assert call_kwargs is not None
    bound_params = call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1]
    assert bound_params.get("doc_type") == "nda"
