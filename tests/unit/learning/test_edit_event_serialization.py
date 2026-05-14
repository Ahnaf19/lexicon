"""Every EditEvent variant round-trips through JSON (discriminated union correctness)."""

from __future__ import annotations

import json
import uuid

import pytest
from pydantic import TypeAdapter

from app.models.pydantic_models import (
    CategoryReordered,
    ChecklistItem,
    EditEvent,
    EvidenceAdded,
    EvidenceCitation,
    EvidenceCorrected,
    ItemAdded,
    ItemRemoved,
    ItemRenamed,
    ItemTextRewritten,
    RequiredToggled,
    StatusChanged,
)

_ITEM_ID = uuid.uuid4()
_CITATION_ID = uuid.uuid4()
_CHUNK_ID = uuid.uuid4()
_DOC_ID = uuid.uuid4()

_CITATION = EvidenceCitation(
    citation_id=_CITATION_ID,
    chunk_id=_CHUNK_ID,
    doc_id=_DOC_ID,
    page_number=1,
    char_offset_start=0,
    char_offset_end=10,
    snippet="test snippet",
    retrieval_score=0.9,
)

_ITEM = ChecklistItem(
    id=uuid.uuid4(),
    category="Parties",
    title="Buyer identified",
    description="Verify buyer identity",
    status="unclear",
    required=True,
    evidence=[],
    confidence=0.0,
    rationale="No evidence yet",
)

_ADAPTER: TypeAdapter[EditEvent] = TypeAdapter(EditEvent)


@pytest.mark.parametrize(
    "event",
    [
        ItemAdded(item=_ITEM),
        ItemRemoved(item_id=_ITEM_ID, reason="false positive"),
        ItemRenamed(item_id=_ITEM_ID, old_title="Old", new_title="New"),
        StatusChanged(item_id=_ITEM_ID, old_status="present", new_status="unclear"),
        EvidenceAdded(item_id=_ITEM_ID, evidence=_CITATION),
        EvidenceCorrected(
            item_id=_ITEM_ID, old_evidence=_CITATION, new_evidence=_CITATION
        ),
        CategoryReordered(
            item_id=_ITEM_ID, old_category="Parties", new_category="Other"
        ),
        ItemTextRewritten(
            item_id=_ITEM_ID, field="rationale", old_text="old", new_text="new"
        ),
        RequiredToggled(item_id=_ITEM_ID, old=True, new=False),
    ],
    ids=[
        "item_added",
        "item_removed",
        "item_renamed",
        "status_changed",
        "evidence_added",
        "evidence_corrected",
        "category_reordered",
        "item_text_rewritten",
        "required_toggled",
    ],
)
def test_round_trip(event: EditEvent) -> None:
    raw = json.loads(json.dumps(event.model_dump(mode="json"), default=str))
    recovered = _ADAPTER.validate_python(raw)
    assert recovered.event_type == event.event_type
    assert recovered.model_dump(mode="json") == event.model_dump(mode="json")
