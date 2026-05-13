"""All domain Pydantic v2 models — single source of truth for API contracts and LLM outputs."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Supporting types for DocumentMeta
# ---------------------------------------------------------------------------


class Party(BaseModel):
    name: str
    role: str | None = None


class MonetaryAmount(BaseModel):
    amount: Decimal
    currency: str
    context: str | None = None


class DefinedTerm(BaseModel):
    term: str
    definition: str
    page: int | None = None


class SignatureBlock(BaseModel):
    signatory: str
    party: str | None = None
    page: int
    signed: bool


# ---------------------------------------------------------------------------
# DocumentMeta (PRD §5b)
# ---------------------------------------------------------------------------


class DocumentMeta(BaseModel):
    doc_id: UUID
    doc_type: Literal[
        "NDA",
        "employment",
        "real_estate_closing",
        "loan_agreement",
        "discovery_production",
        "unknown",
    ]
    parties: list[Party]
    effective_date: date | None = None
    monetary_terms: list[MonetaryAmount]
    defined_terms: list[DefinedTerm]
    exhibits_referenced: list[str]
    signature_blocks: list[SignatureBlock]
    governing_law: str | None = None
    confidence: float


# ---------------------------------------------------------------------------
# EvidenceCitation (PRD §5d)
# ---------------------------------------------------------------------------


class EvidenceCitation(BaseModel):
    citation_id: UUID
    chunk_id: UUID
    doc_id: UUID
    page_number: int
    char_offset_start: int
    char_offset_end: int
    snippet: str
    retrieval_score: float
    rerank_score: float | None = None


# ---------------------------------------------------------------------------
# ChecklistItem + Checklist (PRD §5e)
# ---------------------------------------------------------------------------


class ChecklistItem(BaseModel):
    id: UUID
    category: Literal[
        "Parties",
        "Financial Terms",
        "Required Exhibits",
        "Signatures",
        "Deadlines",
        "Consents",
        "Disclosures",
        "Other",
    ]
    title: str
    description: str
    status: Literal["present", "missing", "unclear"]
    required: bool
    evidence: list[EvidenceCitation] = []
    confidence: float
    rationale: str
    source_template_item_id: UUID | None = None
    learned_from_pattern_ids: list[UUID] = []

    @model_validator(mode="after")
    def evidence_required_when_present(self) -> ChecklistItem:
        if self.status == "present" and not self.evidence:
            raise ValueError("evidence must be non-empty when status is 'present'")
        return self


class Checklist(BaseModel):
    id: UUID
    case_id: UUID
    template_id: UUID
    items: list[ChecklistItem]
    generated_at: datetime
    model_version: str
    prompt_version: str
    eval_metrics: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# LearnedPattern (PRD §5f Layer 2)
# ---------------------------------------------------------------------------


class LearnedPattern(BaseModel):
    id: UUID
    pattern_type: Literal[
        "rename_rule",
        "template_addition",
        "template_removal",
        "status_default",
        "style_preference",
        "category_remap",
    ]
    doc_type_scope: str
    rule_json: dict[str, object]
    supporting_edit_ids: list[UUID]
    confidence: float
    corroborating_edit_count: int
    promoted: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# EditEvent discriminated union (PRD §5f Layer 1) — 9 variants
# ---------------------------------------------------------------------------


class ItemAdded(BaseModel):
    event_type: Literal["item_added"] = "item_added"
    item_payload: dict[str, object]


class ItemRemoved(BaseModel):
    event_type: Literal["item_removed"] = "item_removed"
    item_id: UUID
    reason: str | None = None


class ItemRenamed(BaseModel):
    event_type: Literal["item_renamed"] = "item_renamed"
    item_id: UUID
    old_title: str
    new_title: str


class StatusChanged(BaseModel):
    event_type: Literal["status_changed"] = "status_changed"
    item_id: UUID
    old_status: Literal["present", "missing", "unclear"]
    new_status: Literal["present", "missing", "unclear"]


class EvidenceAdded(BaseModel):
    event_type: Literal["evidence_added"] = "evidence_added"
    item_id: UUID
    evidence: EvidenceCitation


class EvidenceCorrected(BaseModel):
    event_type: Literal["evidence_corrected"] = "evidence_corrected"
    item_id: UUID
    old_evidence: EvidenceCitation
    new_evidence: EvidenceCitation


class CategoryReordered(BaseModel):
    event_type: Literal["category_reordered"] = "category_reordered"
    item_id: UUID
    old_category: str
    new_category: str
    position: int | None = None


class ItemTextRewritten(BaseModel):
    event_type: Literal["item_text_rewritten"] = "item_text_rewritten"
    item_id: UUID
    field: str
    old_text: str
    new_text: str


class RequiredToggled(BaseModel):
    event_type: Literal["required_toggled"] = "required_toggled"
    item_id: UUID
    old: bool
    new: bool


EditEvent = Annotated[
    ItemAdded
    | ItemRemoved
    | ItemRenamed
    | StatusChanged
    | EvidenceAdded
    | EvidenceCorrected
    | CategoryReordered
    | ItemTextRewritten
    | RequiredToggled,
    Field(discriminator="event_type"),
]
