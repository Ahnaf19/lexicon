"""upload_document must return a doc_id that corresponds to a real persisted Document row.

Uses the upload_document handler directly (not HTTP) with a FakeSession that records the
inserted Document. Asserts that UploadResponse.document_id matches the inserted row's id.

The full HTTP-roundtrip integration test (real Postgres + SAVEPOINTS) is deferred to
Phase 4 test scaffolding.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.documents import UploadResponse, upload_document
from app.models.sqlalchemy_models import Document


class _CapturingSession:
    """Records session.add() calls; simulates an empty DB (no existing rows)."""

    def __init__(self) -> None:
        self.added: dict[str, list[Any]] = defaultdict(list)
        self._committed = False

    def add(self, obj: Any) -> None:
        self.added[type(obj).__name__].append(obj)

    async def commit(self) -> None:
        self._committed = True

    async def get(self, model: Any, pk: Any) -> Any:
        return None

    async def execute(self, stmt: Any, params: Any = None) -> Any:
        result = MagicMock()
        # No existing sha256 match → first() returns None
        result.scalars.return_value.first.return_value = None
        result.scalar_one.return_value = 0
        return result

    async def __aenter__(self) -> "_CapturingSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_upload_response_document_id_matches_inserted_row() -> None:
    """UploadResponse.document_id must equal the id of the Document row that was inserted."""
    session = _CapturingSession()

    # Stub out the background ingestion — we only care about what the endpoint returns
    stub_bg = MagicMock()
    stub_bg.add_task = MagicMock()

    # Build a minimal UploadFile stub
    fake_file = MagicMock()
    fake_file.content_type = "application/pdf"
    fake_file.filename = "test_contract.pdf"
    # Minimal valid-ish PDF bytes (just enough to hash)
    fake_file.read = AsyncMock(return_value=b"%PDF-1.4 fake content for testing")

    response: UploadResponse = await upload_document(
        file=fake_file,
        case_id=uuid.UUID(int=0),
        background_tasks=stub_bg,
        session=session,  # type: ignore[arg-type]
    )

    inserted_docs: list[Document] = session.added.get("Document", [])
    assert len(inserted_docs) == 1, (
        "upload_document must insert exactly one Document row"
    )
    inserted_id: uuid.UUID = inserted_docs[0].id
    assert response.document_id == inserted_id, (
        f"UploadResponse.document_id {response.document_id} does not match "
        f"inserted Document.id {inserted_id}"
    )
    assert response.status == "queued"
    assert inserted_docs[0].status == "queued"
