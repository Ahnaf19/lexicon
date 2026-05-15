# Data Flow

This document traces a single checklist generation request end-to-end — from the HTTP call through every LangGraph node to the persisted result. For component descriptions see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Prerequisites

A case must already have at least one indexed document. Documents reach `status=indexed` after the ingestion pipeline completes (see the ingestion section of [ARCHITECTURE.md](./ARCHITECTURE.md#stage-1--ingestion)).

---

## Step 1 — Client sends `POST /checklists/generate`

```bash
curl -N -X POST http://localhost:8000/checklists/generate \
  -H "Content-Type: application/json" \
  -d '{"case_id": "00000000-0000-0000-0000-000000000001", "template_slug": "commercial_contract"}'
```

The `-N` flag disables curl buffering — required for SSE streams.

**What the server does immediately:**
1. Validates the request body against `GenerateRequest` (`app/api/checklists.py:47`).
2. Builds an initial `ChecklistState` with `case_id` and optional `template_slug`.
3. Returns a `text/event-stream` response and begins streaming events as graph nodes fire.

---

## Step 2 — SSE event stream

The client receives a stream of `event: / data:` pairs. Each pair is a JSON object:

```
event: node_start
data: {"node": "classify_doc_set"}

event: node_end
data: {"node": "classify_doc_set"}

event: node_start
data: {"node": "load_template"}

event: node_end
data: {"node": "load_template"}

event: node_start
data: {"node": "retrieve_evidence"}
...

event: done
data: {"status": "complete", "checklist_id": "f612138f-..."}
```

On `event: error`:
```
event: error
data: {"detail": "Template 'foo' not found"}
```

The Streamlit UI (`ui/views/overview.py`) consumes this stream inside `st.status(...)`, showing a live `▸ node_name` / `✓ node_name` log as events arrive.

---

## Step 3 — Graph execution

The LangGraph state machine (`app/generation/graph.py`) processes nodes in this order:

```
warmup_node
    └─▶ classify_doc_set
             └─▶ load_template
                      └─▶ [router: items remain?]
                                  ├─ YES ─▶ retrieve_evidence
                                  │              └─▶ draft_item
                                  │                     └─▶ validate_item
                                  │                              └─▶ critique
                                  │                                     └─▶ [router again]
                                  └─ NO ──▶ assemble
```

### State shape flowing between nodes

```python
{
    "case_id": UUID,
    "template_slug": str | None,
    "template": ChecklistTemplate,          # set by load_template
    "item_index": int,                      # cursor into template.items
    "current_item_slug": str | None,        # slug of item being processed
    "draft_items": list[DraftChecklistItem],# accumulated as items finish critique
    "errors": list[str],
}
```

### Per-item processing detail

**`retrieve_evidence`**
- Embeds the item's `search_query` via Ollama `nomic-embed-text`.
- Runs cosine (pgvector HNSW) + BM25 (tsvector) search in parallel.
- Fuses results via RRF (k=60), takes top-K (default 8) hits.
- Expands each winning window to its parent section.
- Appends a `SearchHit` list to state.

**`draft_item`**
- Builds a prompt with: item title, description, parent-section context texts, few-shot examples (retrieved from `few_shot_examples` table by cosine similarity to this item).
- Calls the LLM (Groq or Ollama) with a JSON-schema-constrained output.
- Parses the result into a `DraftChecklistItem`.

**`validate_item`**
- Runs V1–V5 invariant checks (see [ARCHITECTURE.md](./ARCHITECTURE.md#v1v5-validation-invariants)).
- Hallucinated citations are dropped; ambiguous evidence forces status → `"unclear"`.

**`critique`**
- Applies promoted `LearnedPattern` rules: renames, status defaults, style preferences, category remaps.
- No LLM call — pure rule application from the pattern store.

---

## Step 4 — `assemble` writes to DB

After the router exhausts all items, `assemble`:
1. Persists a `Checklist` row with `case_id`, `generated_at`, `model_version`, `prompt_version`.
2. For each item: inserts a `ChecklistItem` row + one `EvidenceCitation` row per evidence entry.
3. Returns the `checklist_id` in the final `done` SSE event.

All writes happen inside a single async SQLAlchemy session. If any write fails, the whole transaction rolls back and an `error` event is emitted.

---

## Step 5 — Client reads `GET /checklists/{id}`

```bash
curl http://localhost:8000/checklists/f612138f-38cf-45ac-a070-9271b6350ed3
```

Response (abbreviated):

```json
{
  "checklist_id": "f612138f-...",
  "case_id": "00000000-...",
  "generated_at": "2026-05-14T22:29:07Z",
  "model_version": "llama-3.3-70b-versatile",
  "items": [
    {
      "id": "...",
      "title": "Term and duration",
      "category": "Deadlines",
      "status": "present",
      "confidence": 0.94,
      "rationale": "Section 11.1 specifies an initial term of one year...",
      "evidence": [
        {
          "citation_id": "...",
          "page_number": 10,
          "char_offset_start": 4821,
          "char_offset_end": 4973,
          "snippet": "11.1 This agreement begins on the Commencement Date...",
          "retrieval_score": 0.91
        }
      ],
      "learned_from_pattern_ids": []
    }
  ]
}
```

---

## Step 6 — Operator edits + finalize (learning loop)

Operator actions produce `EditEvent` rows automatically:

| HTTP call | Event type recorded |
|---|---|
| `PATCH /checklists/{id}/items/{item_id}` (title change) | `field_edit` |
| `PATCH /checklists/{id}/items/{item_id}` (status change) | `status_change` |
| `POST /checklists/{id}/items` | `item_added` |
| `DELETE /checklists/{id}/items/{item_id}` | `item_removed` |
| `PATCH .../evidence` (add) | `evidence_added` |
| `PATCH .../evidence` (correct) | `evidence_corrected` |
| `PATCH .../evidence` (remove) | `evidence_removed` |

`POST /checklists/{id}/finalize` enqueues `extract_patterns` as a background task. That job reads the `EditEvent` rows, makes one LLM call to distil patterns, and inserts `LearnedPattern` rows. Patterns that reach `corroboration ≥ 3` are promoted and applied on the next generation.

See [EVALUATION.md](./EVALUATION.md) for measured loop closure results.
