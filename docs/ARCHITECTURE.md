# Architecture

Lexicon is a four-stage pipeline: **ingest → retrieve → generate → learn**. Each stage is a self-contained module with typed inputs and outputs; nothing leaks across stage boundaries except the UUIDs of persisted rows.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION                                                      │
│  PDFs / JPGs ──▶ Marker (Surya OCR) ──▶ blocks                  │
│                  ├─ confidence < 0.6 ──▶ TrOCR fallback         │
│                  └─ structured extraction (LLM)                 │
│  Output: Document → Page → Chunk rows (with full provenance)    │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  RETRIEVAL                                                      │
│  Blocks ──▶ section-aware chunking ──▶ windows + parent sections│
│             ──▶ nomic-embed-text (768d, HNSW cosine)            │
│             ──▶ tsvector (BM25 via ts_rank_cd)                  │
│  Query ──▶ dense ‖ sparse ──▶ RRF (k=60) ──▶ parent expansion   │
│  Output: SearchHit list with context_text + provenance          │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  GENERATION (LangGraph state machine)                           │
│  warmup → classify_doc_set → load_template → router             │
│  ├─[items remain]─▶ retrieve_evidence → draft_item              │
│  │                   → validate_item (V1–V5) → critique → router│
│  └─[exhausted]────▶ assemble                                    │
│  Output: Checklist + EvidenceCitation rows                      │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LEARNING (edit loop)                                           │
│  PATCH / POST / DELETE ──▶ typed EditEvent rows                 │
│  POST /finalize ──▶ pattern_extractor (1 LLM call)              │
│      ──▶ LearnedPattern (promoted at corroboration ≥ 3)         │
│      ──▶ few_shot_examples (retrieved at draft time)            │
│  Next generation: load_template + critique consume patterns     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — Ingestion

**Entry point:** `app/ingestion/orchestrator.py`

Documents are submitted via `POST /documents/upload`. Ingestion runs as a FastAPI `BackgroundTask` and transitions the document through `pending → processing → indexed` (or `error`).

### OCR pipeline

| Engine                                      | When it fires                               | Source                 |
| ------------------------------------------- | ------------------------------------------- | ---------------------- |
| Marker (Surya)                              | Primary — all documents                     | `app/ingestion/ocr.py` |
| TrOCR (`microsoft/trocr-large-handwritten`) | Fallback on any block with confidence < 0.6 | `app/ingestion/ocr.py` |

Marker handles native PDFs (text extraction), scanned PDFs, and images in one code path — it auto-detects whether a page has extractable text. Every block carries `ocr_engine`, `confidence`, `bbox`, and `page_number` provenance.

### Structured extraction

After OCR, an LLM call extracts `DocumentMeta` (doc_type, parties, dates) using a constrained-JSON schema. This populates the `Document.doc_type` field used by template auto-detection.

**Key tables:** `documents`, `pages`, `chunks` (SQLAlchemy models in `app/models/sqlalchemy_models.py`)

---

## Stage 2 — Retrieval

**Entry point:** `app/retrieval/retriever.py`

### Chunking

Chunks are stored at two granularities:

- **Window chunks** (~512 tokens): the retrieval unit. Dense and sparse indexes operate at this level.
- **Parent sections** (up to ~3,500 tokens): the generation unit. Retrieved after the window ranking step.

### Hybrid search

Each checklist item query runs two searches in parallel, then fuses the results:

```
item query
    ├─▶ pgvector HNSW cosine (dense, nomic-embed-text 768d)
    └─▶ tsvector ts_rank_cd  (sparse, BM25-approximate)
         └──────────────────────────────────────────────▶ RRF (k=60)
                                                          └─▶ parent expansion
                                                               └─▶ SearchHit
```

RRF (Reciprocal Rank Fusion) with k=60 blends the two ranking lists without requiring score normalisation. Each `SearchHit` carries both the precise window span (for citation anchoring) and the full parent section text (for generation context).

**Key files:**

- `app/retrieval/retriever.py` — main `HybridRetriever` class
- `app/retrieval/embedder.py` — Ollama `nomic-embed-text` wrapper

---

## Stage 3 — Generation

**Entry point:** `app/generation/graph.py`

### LangGraph nodes

| Node                | Responsibility                                                                                                                                        |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `warmup_node`       | Pre-warms the LLM connection to avoid cold-start on the first item                                                                                    |
| `classify_doc_set`  | Classifies the case into a doc_type (e.g. `commercial_contract`) if no template_slug was provided                                                     |
| `load_template`     | Loads the registered `ChecklistTemplate` for the doc_type; applies any promoted `LearnedPattern` mutations (template_addition, template_removal)      |
| `retrieve_evidence` | Runs hybrid search for the current item; returns top-K `SearchHit`s                                                                                   |
| `draft_item`        | LLM call producing a `DraftChecklistItem` with status, rationale, and evidence citations; few-shot examples from the learning store are injected here |
| `validate_item`     | V1–V5 invariant gates (see below)                                                                                                                     |
| `critique`          | Applies promoted `rename_rule`, `status_default`, `style_preference`, `category_remap` patterns; ensures the final item matches operator preferences  |
| `assemble`          | Merges all drafted items into a final `Checklist`; persists to DB                                                                                     |

### V1–V5 validation invariants

| Rule | What it checks                                                   | Failure action              |
| ---- | ---------------------------------------------------------------- | --------------------------- |
| V1   | `status="present"` requires `evidence ≠ []`                      | Coerce status → `"unclear"` |
| V2   | Every citation `page_number` must exist in the ingested document | Drop the citation           |
| V3   | `[doc=X p.N]` or `[doc=X page=N]` reference format in rationale  | Strip malformed refs        |
| V4   | Citation `char_offset_start < char_offset_end`                   | Drop the citation           |
| V5   | Snippet must be non-empty                                        | Drop the citation           |

Hallucinated cites are dropped or coerced — never silently passed through.

**Key files:**

- `app/generation/graph.py` — node wiring
- `app/generation/nodes/` — one file per node
- `app/generation/state.py` — `ChecklistState` TypedDict

---

## Stage 4 — Learning

**Entry point:** `app/learning/`

### Edit capture

Every operator mutation to a checklist (`PATCH`, `POST /items`, `DELETE /items`, evidence mutations) is captured as a typed `EditEvent` row with fields `event_type`, `item_id`, `actor`, `old_value`, `new_value`. The middleware is at `app/learning/edit_capture.py`.

### Pattern extraction

`POST /checklists/{id}/finalize` triggers a background `extract_patterns` job. A single LLM call analyses the accumulated `EditEvent` rows and produces zero or more `LearnedPattern` candidates across six types:

| Pattern type        | What it encodes                                    |
| ------------------- | -------------------------------------------------- |
| `rename_rule`       | Item title should be renamed to a preferred form   |
| `template_addition` | An item should always be present for this doc type |
| `template_removal`  | An item should be removed from future checklists   |
| `status_default`    | An item should default to a particular status      |
| `style_preference`  | Rationale phrasing preference                      |
| `category_remap`    | Item belongs in a different category               |

### Promotion gate

A `LearnedPattern` is promoted (`promoted=True`) only when:

- `corroborating_edit_count ≥ 3`
- `confidence ≥ 0.7`

Promoted patterns are injected automatically on the next checklist generation via `load_template` (template mutations) and `critique` (rename/style/status rules).

**Key files:**

- `app/learning/edit_capture.py` — edit event recorder
- `app/learning/pattern_extractor.py` — LLM-based pattern extraction
- `app/models/sqlalchemy_models.py` — `EditEvent`, `LearnedPattern` ORM models

---

## Infrastructure

| Component              | Role                                                                                | Config                                |
| ---------------------- | ----------------------------------------------------------------------------------- | ------------------------------------- |
| Postgres 16 + pgvector | Primary store for all tables; HNSW index for dense vectors; GIN index for tsvectors | Port 5433 (host), 5432 (container)    |
| Ollama                 | Local embedding inference (`nomic-embed-text`) + optional local LLM (`qwen3:8b`)    | Port 11434                            |
| Groq                   | Primary LLM provider (`llama-3.3-70b-versatile`) — cloud, ~400 TPS                  | `GROQ_API_KEY` in `.env`              |
| FastAPI                | REST + SSE API surface                                                              | Port 8000                             |
| Streamlit              | Operator UI                                                                         | Port 8501                             |
| Langfuse               | LLM/graph observability via LangChain callback                                      | Port 3000 (optional, `--profile obs`) |

See [SETUP.md](./SETUP.md) for how to start each service. See [DATA_FLOW.md](./DATA_FLOW.md) for a request-level walk-through.
