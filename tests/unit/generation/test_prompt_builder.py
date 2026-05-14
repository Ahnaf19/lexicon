"""Tests that the draft_item prompt includes context_text (not snippet) and schema JSON."""

from __future__ import annotations

import json
import uuid

import pytest

from app.generation.nodes.draft_item import _build_evidence_blocks, _PROMPT_TEMPLATE
from app.generation.state import DraftChecklistItem
from app.generation.templates.commercial_contract import COMMERCIAL_CONTRACT
from app.retrieval.hybrid_search import SearchHit

_TEMPLATE = COMMERCIAL_CONTRACT
_PARTIES_ITEM = next(i for i in _TEMPLATE.items if i.slug == "parties")


def _hit(context_text: str, snippet: str = "short snippet") -> SearchHit:
    return SearchHit(
        citation_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        doc_id=uuid.uuid4(),
        page_number=2,
        char_offset_start=0,
        char_offset_end=50,
        snippet=snippet,
        context_text=context_text,
        retrieval_score=0.75,
    )


def test_evidence_blocks_use_context_text_not_snippet():
    ctx = "Full parent-section text spanning several sentences about party definitions."
    snip = "short"
    h = _hit(context_text=ctx, snippet=snip)
    blocks = _build_evidence_blocks([h])
    assert ctx in blocks
    assert snip not in blocks


def test_evidence_blocks_tagged_with_ei_labels():
    h1 = _hit("Section one content.")
    h2 = _hit("Section two content.")
    blocks = _build_evidence_blocks([h1, h2])
    assert "[E1 doc=" in blocks
    assert "[E2 doc=" in blocks


def test_evidence_blocks_empty_returns_placeholder():
    blocks = _build_evidence_blocks([])
    assert "No evidence retrieved" in blocks


def test_prompt_contains_schema():
    schema_json = json.dumps(DraftChecklistItem.model_json_schema(), indent=2)
    h = _hit("Some context text.")
    evidence_blocks = _build_evidence_blocks([h])
    prompt = _PROMPT_TEMPLATE.format(
        schema=schema_json,
        item_title=_PARTIES_ITEM.title,
        item_description=_PARTIES_ITEM.description,
        item_category=_PARTIES_ITEM.category,
        item_required=str(_PARTIES_ITEM.required),
        learned_patterns_json="[]",
        few_shot_pairs_json="[]",
        evidence_blocks=evidence_blocks,
    )
    assert "status" in prompt  # schema field present
    assert "cited_evidence" in prompt
    assert _PARTIES_ITEM.title in prompt
    assert "No evidence retrieved" not in prompt  # we gave it a hit


def test_prompt_contains_template_item_details():
    schema_json = json.dumps(DraftChecklistItem.model_json_schema(), indent=2)
    evidence_blocks = _build_evidence_blocks([])
    prompt = _PROMPT_TEMPLATE.format(
        schema=schema_json,
        item_title=_PARTIES_ITEM.title,
        item_description=_PARTIES_ITEM.description,
        item_category=_PARTIES_ITEM.category,
        item_required=str(_PARTIES_ITEM.required),
        learned_patterns_json="[]",
        few_shot_pairs_json="[]",
        evidence_blocks=evidence_blocks,
    )
    assert _PARTIES_ITEM.title in prompt
    assert _PARTIES_ITEM.description in prompt
    assert _PARTIES_ITEM.category in prompt
