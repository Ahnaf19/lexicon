"""TemplateItem and ChecklistTemplate — the code-level template types for generation (PRD §5e)."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel

_CATEGORY = Literal[
    "Parties",
    "Financial Terms",
    "Required Exhibits",
    "Signatures",
    "Deadlines",
    "Consents",
    "Disclosures",
    "Other",
]

# Namespace for deterministic template UUIDs; stable across runs.
_TEMPLATE_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL


class TemplateItem(BaseModel):
    """One item in a checklist template, carrying the retrieval sub-query."""

    slug: str
    title: str
    description: str
    category: _CATEGORY
    required: bool
    sub_query: str

    def stable_uuid(self, template_id: uuid.UUID) -> uuid.UUID:
        """Deterministic UUID derived from template + slug; used as source_template_item_id."""
        return uuid.uuid5(template_id, self.slug)


class ChecklistTemplate(BaseModel):
    """A code-defined checklist template with its canonical items."""

    slug: str
    id: uuid.UUID
    name: str
    doc_type: str
    version: str = "v1"
    items: list[TemplateItem]

    @classmethod
    def make(cls, slug: str, name: str, doc_type: str, items: list[TemplateItem]) -> "ChecklistTemplate":
        """Build with a deterministic id derived from the slug."""
        return cls(
            slug=slug,
            id=uuid.uuid5(_TEMPLATE_NS, f"lexicon.template/{slug}"),
            name=name,
            doc_type=doc_type,
            items=items,
        )
