# Lexicon

Grounded document-checklist generator for messy legal documents.
Upload a set of case documents → get a checklist of required items with evidence citations → edit it → the system learns from your edits.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Client Layer                           │
│                                                                 │
│   ┌──────────────────┐          ┌──────────────────────────┐   │
│   │  Streamlit UI    │          │  REST / SSE (FastAPI)    │   │
│   │  :8501           │◄────────►│  :8000                   │   │
│   └──────────────────┘          └────────────┬─────────────┘   │
└────────────────────────────────────────────── │ ────────────────┘
                                                │
          ┌─────────────────────────────────────┼──────────────────────┐
          │                                     │                      │
          ▼                                     ▼                      ▼
  ┌───────────────┐                   ┌─────────────────┐    ┌────────────────┐
  │   Ingestion   │                   │   Generation    │    │   Learning     │
  │   Pipeline    │                   │  (LangGraph)    │    │   Pipeline     │
  └───────┬───────┘                   └────────┬────────┘    └───────┬────────┘
          │                                    │                     │
          ▼                                    ▼                     ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                         Postgres 16 + pgvector                            │
  │                                                                           │
  │  documents  pages  chunks          checklists  checklist_items            │
  │  ┌────────────────────────┐        ┌──────────────────────────────────┐  │
  │  │ embedding  vector(768) │        │ evidence_citations               │  │
  │  │ tsv        tsvector    │        │ edit_events  (append-only)       │  │
  │  │ HNSW (cosine)  GIN     │        │ learned_patterns  few_shot_examples│ │
  │  └────────────────────────┘        └──────────────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────────────┘
          │                                    │
          ▼                                    ▼
  ┌───────────────┐                   ┌─────────────────┐
  │  Ollama       │                   │   Langfuse      │
  │  nomic-embed  │                   │   (tracing)     │
  │  :11434       │                   │   :3000         │
  └───────────────┘                   └─────────────────┘
          │
  ┌───────┴───────┐
  │  Groq API     │  ← primary LLM (llama-3.3-70b / llama-3.1-8b)
  │  or           │
  │  Ollama LLM   │  ← fallback (qwen3:8b)
  └───────────────┘
```

---

## Document Ingestion Flow

```
POST /documents/upload
        │
        ▼
  sha256(file) ──► already exists? ──► short-circuit (idempotent)
        │ new
        ▼
  Marker OCR  ──────────────────────────────────────────────────────┐
  (primary)                                                         │
        │ low confidence / handwriting detected                     │
        ▼                                                           │
  TrOCR fallback                                                    │
        │                                                           │
        └──────────────────────────────────────────────────────────►│
                                                                    ▼
                                                     Structured extraction
                                                     (DocumentMeta: parties,
                                                      dates, monetary terms,
                                                      sig blocks, doc_type)
                                                                    │
                                                                    ▼
                                                          Chunking (semantic +
                                                           structural boundaries)
                                                                    │
                                                                    ▼
                                                     nomic-embed-text → vector(768)
                                                     tsvector GENERATED ALWAYS AS
                                                                    │
                                                                    ▼
                                                        INSERT chunks (pgvector)
                                                        HNSW + GIN indexes
```

---

## Checklist Generation (LangGraph Pipeline)

```
POST /checklists/generate  →  SSE stream of node events
        │
        ▼
┌───────────────────┐
│  classify_doc_set │  DocumentMeta → pick checklist template
└────────┬──────────┘
         │
         ▼
┌────────────────────────┐
│  load_template +       │  active checklist template +
│  learned_patterns      │  promoted patterns (confidence ≥ 0.7, n ≥ 3)
└────────┬───────────────┘
         │
         │  for each template item  (asyncio.gather, capped concurrency)
         ▼
┌────────────────────────┐
│  retrieve_evidence     │  hybrid search: HNSW dense + GIN tsvector → RRF fusion
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  draft_item            │  LLM (quality model) + CIPHER few-shot:
│                        │  top-3 (original_draft → edited_final) from
│                        │  few_shot_examples, filtered by doc_type
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  validate_item         │  cited pages exist in source doc?
│                        │  status="present" → evidence non-empty?
│                        │  hallucinated citation → coerce to "unclear"
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  critique              │  apply active learned_patterns, rewrite on violation
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  assemble_checklist    │  → Checklist with evidence citations
└────────────────────────┘
```

---

## Edit Loop & Learning

```
Operator reviews checklist in UI
        │
        │  PATCH /checklists/{id}/items/{item_id}  (status, evidence, title, …)
        ▼
  edit_events  (append-only, 9 event types)
  ┌────────────────────────────────────────────┐
  │ item_added      item_removed               │
  │ item_renamed    status_changed             │
  │ evidence_added  evidence_corrected         │
  │ category_reordered  item_text_rewritten    │
  │ required_toggled                           │
  └──────────────────┬─────────────────────────┘
                     │
  POST /checklists/{id}/finalize
                     │
                     ▼
           pattern_extractor
           (BackgroundTask)
                     │
                     ▼
        corroborating_edit_count++
                     │
          ┌──────────┴──────────┐
          │ count ≥ 3 AND       │
          │ confidence ≥ 0.7    │
          └──────────┬──────────┘
                     │ promote
                     ▼
           learned_patterns
           ┌──────────────────────────────┐
           │ rename_rule                  │
           │ template_addition/removal    │  ──► mutates active template
           │ status_default               │
           │ style_preference             │
           │ category_remap               │
           └──────────────────────────────┘
                     │
                     ▼
           few_shot_examples  (CIPHER bank)
           used in next draft_item node
```

---

## Key Invariants

| Rule | Where enforced |
|------|---------------|
| `status="present"` requires non-empty `evidence` | `ChecklistItem` Pydantic validator + `validate_item` node |
| Hallucinated citation → coerce to `"unclear"`, never `"present"` | `validate_item` node |
| Embeddings always local (`nomic-embed-text`) regardless of LLM provider | `app/core/config.py` |
| Edit events are append-only | Convention; no UPDATE/DELETE on `edit_events` |
| LLM provider swap is one env var (`LLM_PROVIDER=groq\|ollama`) | `app/core/llm.py` |

---

## Stack

| Concern | Choice |
|---------|--------|
| API | FastAPI + SSE (`sse-starlette`) |
| State machine | LangGraph 1.x |
| LLM (primary) | Groq `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` |
| LLM (fallback) | Ollama `qwen3:8b` / `llama3.1:8b` |
| Embeddings | Ollama `nomic-embed-text` 768-dim |
| OCR | Marker (primary) · TrOCR (handwriting) |
| DB | Postgres 16 + pgvector · HNSW + GIN |
| Observability | Langfuse (LangChain callback) · loguru |
| UI | Streamlit |
| Env | Python 3.12 · uv · ruff · mypy --strict |
