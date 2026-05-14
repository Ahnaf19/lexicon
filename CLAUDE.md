# Lexicon — Project Conventions

Grounded Document-Checklist Generator for messy legal documents. Take-home, 2-day deadline. Full spec at `docs/planning/PRD.md` — read the relevant section before any non-trivial change.

## Stack

- **Python 3.12**, managed by **uv**. Use `uv add`, `uv remove`, `uv sync`, `uv run`. Never `pip install` directly.
- **FastAPI** + **Pydantic v2** + **SQLAlchemy 2 (async)** + **Alembic**.
- **Postgres 16 + pgvector**. HNSW for dense; `tsvector` + GIN for sparse.
- **LangGraph 1.x** for the generation state machine. Provider abstraction via `langchain.chat_models.init_chat_model`.
- **LLM:** Groq (`llama-3.3-70b-versatile`) primary; Ollama (`qwen3:8b`) fallback. Selected by `settings.llm_provider`.
- **Embeddings:** local Ollama `nomic-embed-text` (768-dim, 8192-ctx).
- **OCR:** Marker (Datalab) primary; TrOCR for handwriting fallback only.
- **Logging:** `loguru` everywhere. `from loguru import logger`. Never `print`, never stdlib `logging`.
- **Observability:** Langfuse via LangChain callback for LLM/graph tracing.
- **Testing:** `pytest` + `pytest-asyncio` (auto mode) + `syrupy`. LLM stubbed via `FakeListChatModel` — never live calls in CI.
- **Lint/type:** `ruff` (lint + format) + `mypy --strict` + `pre-commit`.

## Code conventions

- Pydantic v2 on every module boundary. **No `dict[str, Any]`** in app code.
- All I/O is `async`. LLM and HTTP calls wrapped with `tenacity` retry and `aiolimiter` rate-limit.
- Settings exclusively through `app.core.config.settings`. Never read `os.environ` from business code.
- Module structure follows PRD §16. Don't reorganize.
- Provenance fields (`doc_id, page, bbox, char_offsets, ocr_engine, confidence`) flow through every transform — never silently dropped.

## Logging style (loguru)

- Bind context at entry: `logger.bind(doc_id=..., checklist_id=..., node=...).info("...")`.
- `INFO` for business events, `DEBUG` for retrieval scores and intermediate state, `WARNING` for handled degradations (e.g. OCR low-confidence), `ERROR` only for true failures.
- JSON serializer in prod, human-readable in dev. Configured once in `app.core.logging`.

## Documentation style

- Short but precise. Every public function has a 1–3-line docstring describing what + why, not how.
- Module-level docstring on each `__init__.py` explaining scope.
- README and PRD references with `§` and section number where applicable.

## Anti-patterns — don't do these

- Don't add Celery, RQ, or Redis. `FastAPI BackgroundTasks` is the PoC choice.
- Don't migrate off pgvector. Qdrant is deferred.
- Don't add LiteLLM. `init_chat_model` is the provider abstraction.
- Don't rearrange LangGraph nodes. The sequence in PRD §5e is intentional.
- Don't fine-tune. Edit-loop improvement is retrieval-augmented few-shot (PRD §5f).
- Don't coerce a hallucinated citation to `"present"`. Always coerce to `"unclear"`.
- Don't introduce a cross-encoder reranker until the core loop is green (deferred list).

## Workflow agreements

- Read the relevant PRD section before changes. Most "should we…" questions have a PRD answer.
- Use the `/commit` skill for commits. Group related changes; never per-file unless they truly diverge.
- Defer scope, don't expand it. Cut list in PRD §14 is binding under time pressure.
- The 25-point edit loop (Task 4) is the differentiator. Protect it from cuts.
- For multi-session work, end with a one-paragraph note in `docs/planning/sessions/YYYY-MM-DD-<topic>.md`.
- At each phase boundary (before commit), invoke relevant subagents to review the diff. Use the pattern "review the phase N implementation by the relevant engineers" — agents are read-heavy and find things the main session misses.

## Subagents

Four specialist agents are available in `.claude/agents/` — invoke them when their scope matches. Each has narrow scope and explicit don'ts.

- `ml-engineer` — generation pipeline, prompts, retrieval scoring, provider abstraction.
- `data-engineer` — OCR, chunking, embeddings, pgvector, ingestion idempotency.
- `qa-engineer` — pytest, fixtures, snapshot tests, eval scripts.
- `product-engineer` — scope/cut decisions and rubric alignment. Invoke BEFORE a major feature and AT END of each phase.

## Planning docs

`docs/planning/` is gitignored — never commit anything inside it. Read for context (PRD lives there). Never modify. If plan changes/updated: extend to new version for planning/prd and old one should point to that.
