# Tests

PRD §6.8.

## Layout

- `tests/unit/<module>/` — per-submodule unit tests.
- `tests/integration/test_pipeline.py` — one end-to-end test against fixture documents with the LLM stubbed.
- `tests/snapshot/` — `syrupy` snapshots of canonical `Checklist` outputs.

## Stub the LLM

Tests must **never** hit Groq or Ollama. Use `langchain_core.language_models.fake_chat_models.FakeListChatModel` returning canned `ChecklistItem` JSON. Patch `app.core.llm.get_chat_model` in a session-scoped `conftest.py` fixture.

## Async

`pytest-asyncio` in auto mode. Use `httpx.AsyncClient` for FastAPI tests, never the sync `TestClient`.

## Postgres fixtures

Spin up a fresh test schema **per session** (not per test — too slow). Use SAVEPOINTS for per-test isolation.

## What's not in tests

The `eval/` directory is separate. It runs against real LLM endpoints, produces the results table (PRD §10), and is **not** part of `make test`. Run via `make eval`.

## Coverage philosophy

Cover invariants, citations, and edit-loop behavior. Skip trivial getters/setters. The integration test is the single most valuable artifact — give it the most polish.
