# API Reference

The OpenAPI spec is auto-generated and available at `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` when the server is running.

> [!NOTE]
> All UUIDs are version 4. Timestamps are ISO 8601 UTC. The base URL below assumes the default local setup; in Docker it's the same port.

```
Base URL: http://localhost:8000
```

---

## Meta

### `GET /healthz`

DB reachability check.

```bash
curl http://localhost:8000/healthz
```

```json
{"status": "ok", "db": "ok"}
```

### `GET /evidence/{citation_id}`

Retrieve a single evidence citation — chunk text, page number, character offsets, retrieval score.

```bash
curl http://localhost:8000/evidence/abc12345-...
```

```json
{
  "citation_id": "abc12345-...",
  "chunk_id": "...",
  "doc_id": "...",
  "page_number": 10,
  "char_offset_start": 4821,
  "char_offset_end": 4973,
  "snippet": "11.1 This agreement begins on the Commencement Date...",
  "retrieval_score": 0.91,
  "rerank_score": null
}
```

---

## Documents

### `POST /documents/upload`

Ingest a PDF or image into a case. Ingestion runs in the background; poll `/documents/{id}/status` for completion.

```bash
curl -X POST "http://localhost:8000/documents/upload?case_id=00000000-0000-0000-0000-000000000001" \
  -F "file=@contract.pdf;type=application/pdf"
```

```json
{"document_id": "d1a2b3c4-...", "status": "pending"}
```

Accepted MIME types: `application/pdf`, `image/jpeg`, `image/png`.

### `GET /documents/{document_id}/status`

Poll ingestion status. Terminal states: `indexed` (success) or `error` / `failed`.

```bash
curl http://localhost:8000/documents/d1a2b3c4-.../status
```

```json
{"status": "indexed", "last_event_at": "2026-05-14T22:10:00Z"}
```

### `GET /documents/{document_id}`

Full document metadata including page count, chunk count, OCR engine, and extracted `DocumentMeta`.

```bash
curl http://localhost:8000/documents/d1a2b3c4-...
```

### `GET /documents/cases`

List all distinct cases with their document counts.

```bash
curl http://localhost:8000/documents/cases
```

```json
[
  {"case_id": "00000000-0000-0000-0000-000000000001", "doc_count": 5}
]
```

### `GET /documents/cases/{case_id}/documents`

List documents in a case, newest first.

```bash
curl http://localhost:8000/documents/cases/00000000-.../documents
```

```json
[
  {
    "document_id": "d1a2b3c4-...",
    "original_filename": "contract.pdf",
    "status": "indexed",
    "doc_type": "commercial_contract",
    "total_pages": 14
  }
]
```

---

## Checklists

### `POST /checklists/generate`

Start checklist generation. Returns a **Server-Sent Events** stream.

```bash
curl -N -X POST http://localhost:8000/checklists/generate \
  -H "Content-Type: application/json" \
  -d '{"case_id": "00000000-...", "template_slug": "commercial_contract"}'
```

`template_slug` is optional — omit it for auto-detection. Valid values: `commercial_contract`, `nda`.

**SSE event sequence:**

| Event | Data | When |
|---|---|---|
| `node_start` | `{"node": "<name>"}` | Before each LangGraph node |
| `node_end` | `{"node": "<name>"}` | After each node completes |
| `done` | `{"status": "complete", "checklist_id": "<uuid>"}` | Generation complete |
| `error` | `{"detail": "<message>"}` | Unrecoverable error |

> [!WARNING]
> The SSE stream does not use `Content-Length`. Clients that buffer the full response body before parsing will hang until the stream closes. Use `curl -N` or an SSE-aware client library.

### `GET /checklists/{checklist_id}`

Retrieve a checklist with all items and evidence citations.

```bash
curl http://localhost:8000/checklists/f612138f-...
```

### `PATCH /checklists/{checklist_id}/items/{item_id}`

Edit an item. Only include fields you want to change. Every changed field generates an `EditEvent` row.

```bash
curl -X PATCH http://localhost:8000/checklists/f612138f-.../items/abc12345-... \
  -H "Content-Type: application/json" \
  -d '{"title": "Governing law and jurisdiction", "status": "present"}'
```

Editable fields: `title`, `description`, `status`, `required`, `confidence`, `rationale`, `category`.

> [!WARNING]
> `status="present"` requires at least one existing evidence citation on the item. The server returns 422 if you attempt to set present on an item with no evidence.

### `POST /checklists/{checklist_id}/items`

Add a new item. The client must supply a UUID for `id`.

```bash
curl -X POST http://localhost:8000/checklists/f612138f-.../items \
  -H "Content-Type: application/json" \
  -d '{
    "id": "bbbb1111-...",
    "title": "Force majeure",
    "category": "Other",
    "status": "missing",
    "required": true,
    "description": "",
    "confidence": 0.5,
    "rationale": "",
    "evidence": [],
    "learned_from_pattern_ids": []
  }'
```

### `DELETE /checklists/{checklist_id}/items/{item_id}`

Remove an item. Returns 204 No Content.

```bash
curl -X DELETE http://localhost:8000/checklists/f612138f-.../items/abc12345-...
```

### `PATCH /checklists/{checklist_id}/items/{item_id}/evidence`

Add, correct, or remove an evidence citation.

```bash
# Add
curl -X PATCH http://localhost:8000/checklists/f612138f-.../items/abc12345-.../evidence \
  -H "Content-Type: application/json" \
  -d '{
    "action": "add",
    "evidence": {
      "citation_id": "cccc2222-...",
      "chunk_id": "...",
      "doc_id": "...",
      "page_number": 3,
      "char_offset_start": 100,
      "char_offset_end": 200,
      "snippet": "The term shall be...",
      "retrieval_score": 0.85
    }
  }'

# Remove
curl -X PATCH http://localhost:8000/checklists/f612138f-.../items/abc12345-.../evidence \
  -H "Content-Type: application/json" \
  -d '{"action": "remove", "old_citation_id": "cccc2222-..."}'
```

`action` values: `add`, `correct`, `remove`.

### `POST /checklists/{checklist_id}/finalize`

Mark the checklist as finalized and trigger background pattern extraction. Returns 202.

```bash
curl -X POST http://localhost:8000/checklists/f612138f-.../finalize
```

### `GET /checklists/cases/{case_id}/checklists`

List checklists generated for a case, newest first.

```bash
curl http://localhost:8000/checklists/cases/00000000-.../checklists
```

### `GET /checklists/learned-patterns`

List learned patterns with optional filters.

```bash
curl "http://localhost:8000/checklists/learned-patterns?promoted=true&limit=20"
```

Query params: `doc_type`, `pattern_type`, `promoted` (bool), `limit` (default 50), `offset` (default 0).

### `POST /checklists/learned-patterns/{pattern_id}/dismiss`

Decrement corroboration count. Demotes the pattern if it falls below the promotion threshold.

```bash
curl -X POST http://localhost:8000/checklists/learned-patterns/dddd3333-.../dismiss
```

---

## Operator identity

Pass `X-Operator-Id: <your-id>` on any mutating request. It's recorded on every `EditEvent` row. Unauthenticated requests default to the actor `"default"`.

```bash
curl -X PATCH ... -H "X-Operator-Id: alice@firm.com" ...
```
