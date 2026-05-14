"""ChecklistState TypedDict and DraftChecklistItem LLM output schema (PRD §5e)."""

from __future__ import annotations

import uuid
from typing import Literal, TypedDict

from pydantic import BaseModel

from app.generation.templates.types import ChecklistTemplate
from app.models.pydantic_models import Checklist, ChecklistItem, LearnedPattern
from app.retrieval.hybrid_search import SearchHit


class DraftChecklistItem(BaseModel):
    """Slim schema for structured LLM output.

    validate_item resolves cited_evidence Ei labels to SearchHit entries and
    builds the final ChecklistItem with proper EvidenceCitation objects.
    """

    status: Literal["present", "missing", "unclear"]
    confidence: float
    rationale: str
    cited_evidence: list[str] = []  # Ei labels e.g. ["E1", "E3"]


class ChecklistState(TypedDict, total=False):
    """Mutable state object threaded through the LangGraph pipeline."""

    case_id: uuid.UUID
    document_ids: list[uuid.UUID]           # filtered to status="indexed" at classify time
    template_slug: str
    template: ChecklistTemplate
    learned_patterns: list[LearnedPattern]  # always [] in phase 4; phase 5 fills in
    items_in_progress: dict[str, ChecklistItem | None]  # keyed by TemplateItem.slug
    search_hits_by_item: dict[str, list[SearchHit]]
    errors: list[dict]                      # type: ignore[type-arg]
    model_version: str
    prompt_version: str
    current_item_slug: str | None           # cursor for the sequential per-item loop
    item_index: int                         # internal loop counter (not exposed via API)
    # assemble output — must be declared for LangGraph to retain across state merge
    checklist: Checklist | None
    checklist_id: uuid.UUID | None
