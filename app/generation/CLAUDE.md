# Generation module — LangGraph pipeline

PRD §5e. Node sequence is intentional; do not rearrange.

## Pipeline

```
classify_doc_set → load_template → [per item] retrieve_evidence → draft_item
                                                                       ↓
                                                              validate_item
                                                                       ↓
                                                                   critique
                                                                       ↓
                                                          assemble_checklist
```

## Hard invariants

- `status="present"` requires `len(evidence) > 0`. Enforced in `validate_item`. Violation → coerce to `"unclear"`.
- Every cited `(doc_id, page_number)` must exist in the retrieved evidence for the item. Violation → drop the citation; re-check the present invariant.
- `"unclear"` is a first-class status. Refuse to guess when evidence is ambiguous.

## LLM access

Always via `app.core.llm.get_chat_model(role="quality"|"fast")`. Never instantiate provider clients directly. JSON-schema-constrained decoding at the provider level. Pydantic validation post-decode with one retry under stricter system prompt.

## Streaming

`POST /checklists/generate` returns `text/event-stream` from `graph.astream_events(version="v2")`. Forward node start/end events plus the in-flight `checklist_item_id` so the UI can highlight per-item progress.

## Templates

In `app/generation/templates/`. PoC ships `commercial_contract.py` and `nda.py`. Each is a `ChecklistTemplate` with a list of canonical items, each carrying a sub-query template string used by retrieval.

## Prompts

In `app/generation/prompts/`. Versioned (`v1`, `v2`, …) so prompt drift is grep-able. The active version is recorded in `Checklist.prompt_version` on every run.
