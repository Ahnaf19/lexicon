"""Tests for classify_doc_set majority-vote and skip-non-indexed-docs logic."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.generation.nodes.classify_doc_set import classify_doc_set
from app.generation.state import ChecklistState

_CASE_ID = uuid.UUID(int=99)


def _make_state(template_slug: str | None = None) -> ChecklistState:
    s: ChecklistState = {"case_id": _CASE_ID, "document_ids": [], "errors": []}
    if template_slug is not None:
        s["template_slug"] = template_slug
    return s


def _mock_rows(rows: list[tuple]) -> MagicMock:
    """Build a mock async session that returns the given rows from execute."""
    session = AsyncMock()
    result = MagicMock()
    result.all.return_value = rows
    session.execute = AsyncMock(return_value=result)
    return session


def _patch_session(rows: list[tuple]):
    """Context manager that patches SessionLocal to return a mock session."""
    session = _mock_rows(rows)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return patch("app.generation.nodes.classify_doc_set.SessionLocal", return_value=cm)


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_majority_vote_commercial_contract():
    rows = [
        (uuid.uuid4(), "commercial_contract"),
        (uuid.uuid4(), "commercial_contract"),
        (uuid.uuid4(), "commercial_contract"),
        (uuid.uuid4(), "nda"),
    ]
    with _patch_session(rows):
        result = await classify_doc_set(_make_state())
    assert result["template_slug"] == "commercial_contract"
    assert len(result["document_ids"]) == 4


@pytest.mark.asyncio
async def test_majority_vote_nda():
    rows = [
        (uuid.uuid4(), "nda"),
        (uuid.uuid4(), "nda"),
        (uuid.uuid4(), "service"),
    ]
    with _patch_session(rows):
        result = await classify_doc_set(_make_state())
    assert result["template_slug"] == "nda"


@pytest.mark.asyncio
async def test_explicit_slug_overrides_majority():
    rows = [
        (uuid.uuid4(), "nda"),
        (uuid.uuid4(), "nda"),
        (uuid.uuid4(), "nda"),
    ]
    with _patch_session(rows):
        result = await classify_doc_set(_make_state(template_slug="commercial_contract"))
    assert result["template_slug"] == "commercial_contract"


@pytest.mark.asyncio
async def test_unknown_doc_type_falls_back_to_commercial_contract():
    rows = [
        (uuid.uuid4(), "loan_agreement"),
        (uuid.uuid4(), "strategic_alliance"),
    ]
    with _patch_session(rows):
        result = await classify_doc_set(_make_state())
    assert result["template_slug"] == "commercial_contract"


# ---------------------------------------------------------------------------
# Empty indexed set → raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_indexed_raises():
    with _patch_session([]):
        with pytest.raises(ValueError, match="No indexed documents"):
            await classify_doc_set(_make_state())
