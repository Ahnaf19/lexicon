"""HTTP client wrapping all Lexicon API endpoints.

Read methods are wrapped with @st.cache_data(ttl=10) so repeated renders
within the same 10-second window don't hit the API on every Streamlit rerun.
Mutation methods are uncached; callers clear the relevant cache then rerun.
"""
from __future__ import annotations

import os
from typing import Any, Iterator

import httpx
import streamlit as st

from utils import sse_iter_events

BASE_URL: str = (
    os.environ.get("LEXICON_API_URL")
    or os.environ.get("APP_BASE_URL")
    or "http://localhost:8000"
)
_TIMEOUT = 30
_STREAM_TIMEOUT = 300


def _client(**kwargs: Any) -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=_TIMEOUT, **kwargs)


# ─── Cached reads ────────────────────────────────────────────────────────────


@st.cache_data(ttl=10)
def get_cases() -> list[dict]:  # type: ignore[type-arg]
    """GET /documents/cases → list of {case_id, doc_count}."""
    with _client() as c:
        r = c.get("/documents/cases")
        r.raise_for_status()
        return r.json()


@st.cache_data(ttl=10)
def get_case_documents(case_id: str) -> list[dict]:  # type: ignore[type-arg]
    """GET /documents/cases/{id}/documents → list of document summaries."""
    with _client() as c:
        r = c.get(f"/documents/cases/{case_id}/documents")
        r.raise_for_status()
        return r.json()


@st.cache_data(ttl=10)
def get_case_checklists(case_id: str) -> list[dict]:  # type: ignore[type-arg]
    """GET /checklists/cases/{id}/checklists → list of checklist summaries."""
    with _client() as c:
        r = c.get(f"/checklists/cases/{case_id}/checklists")
        r.raise_for_status()
        return r.json()


@st.cache_data(ttl=10)
def get_checklist(checklist_id: str) -> dict:  # type: ignore[type-arg]
    """GET /checklists/{id} → full checklist with items + evidence."""
    with _client() as c:
        r = c.get(f"/checklists/{checklist_id}")
        r.raise_for_status()
        return r.json()


@st.cache_data(ttl=5)
def get_document_status(doc_id: str) -> dict:  # type: ignore[type-arg]
    """GET /documents/{id}/status → {status, last_event_at}."""
    with _client() as c:
        r = c.get(f"/documents/{doc_id}/status")
        r.raise_for_status()
        return r.json()


@st.cache_data(ttl=10)
def get_patterns(
    doc_type: str | None = None,
    promoted: bool | None = None,
    pattern_type: str | None = None,
) -> list[dict]:  # type: ignore[type-arg]
    """GET /checklists/learned-patterns with optional filters."""
    params: dict[str, str] = {}
    if doc_type:
        params["doc_type"] = doc_type
    if promoted is not None:
        params["promoted"] = str(promoted).lower()
    if pattern_type:
        params["pattern_type"] = pattern_type
    with _client() as c:
        r = c.get("/checklists/learned-patterns", params=params)
        r.raise_for_status()
        return r.json()


# ─── Mutations (uncached) ────────────────────────────────────────────────────


def upload_document(
    file_bytes: bytes, filename: str, mime: str, case_id: str
) -> dict:  # type: ignore[type-arg]
    """POST /documents/upload → {document_id, status}."""
    with _client(timeout=120) as c:
        r = c.post(
            f"/documents/upload?case_id={case_id}",
            files={"file": (filename, file_bytes, mime)},
        )
        r.raise_for_status()
        return r.json()


def patch_item(checklist_id: str, item_id: str, fields: dict) -> dict:  # type: ignore[type-arg]
    """PATCH /checklists/{id}/items/{item_id} with non-None fields only."""
    body = {k: v for k, v in fields.items() if v is not None}
    with _client() as c:
        r = c.patch(f"/checklists/{checklist_id}/items/{item_id}", json=body)
        r.raise_for_status()
        return r.json()


def add_item(checklist_id: str, item: dict) -> dict:  # type: ignore[type-arg]
    """POST /checklists/{id}/items with a client-minted UUID on the item."""
    with _client() as c:
        r = c.post(f"/checklists/{checklist_id}/items", json=item)
        r.raise_for_status()
        return r.json()


def delete_item(checklist_id: str, item_id: str) -> None:
    """DELETE /checklists/{id}/items/{item_id}."""
    with _client() as c:
        r = c.delete(f"/checklists/{checklist_id}/items/{item_id}")
        r.raise_for_status()


def finalize_checklist(checklist_id: str) -> dict:  # type: ignore[type-arg]
    """POST /checklists/{id}/finalize → {checklist_id, status}."""
    with _client() as c:
        r = c.post(f"/checklists/{checklist_id}/finalize")
        r.raise_for_status()
        return r.json()


def dismiss_pattern(pattern_id: str) -> dict:  # type: ignore[type-arg]
    """POST /checklists/learned-patterns/{id}/dismiss → updated LearnedPattern."""
    with _client() as c:
        r = c.post(f"/checklists/learned-patterns/{pattern_id}/dismiss")
        r.raise_for_status()
        return r.json()


def stream_generate(
    case_id: str, template_slug: str | None
) -> Iterator[tuple[str, dict]]:  # type: ignore[type-arg]
    """Consume the SSE stream from POST /checklists/generate.

    Yields (event_name, data_dict) tuples. Terminates when a 'done' or
    'error' event is yielded (caller decides how to stop iterating).
    """
    body: dict[str, Any] = {"case_id": case_id}
    if template_slug:
        body["template_slug"] = template_slug

    with httpx.Client(base_url=BASE_URL, timeout=_STREAM_TIMEOUT) as client:
        with client.stream("POST", "/checklists/generate", json=body) as response:
            response.raise_for_status()
            yield from sse_iter_events(response.iter_lines())
