# Ingestion module

OCR + structured extraction. PRD §5a–§5b.

## Tools

- **Marker** (`marker-pdf`): primary OCR + layout. One path handles native PDFs and scans. Emits Markdown + per-block JSON with `(bbox, page, confidence)`.
- **TrOCR** (`microsoft/trocr-large-handwritten`): handwriting fallback. Invoke only on blocks Surya labels `handwriting` or printed-text confidence < 0.5.
- **Regex** for stable fields (dates, money, exhibit refs). **LLM** (via `app.core.llm.get_chat_model(role="fast")`) for fuzzy fields (parties, doc_type, defined terms, governing law).

## Confidence routing

- `mean_block_confidence < 0.80` → set `pages.ocr_confidence_mean` and flag for UI.
- Block `confidence < 0.50` after both Marker and TrOCR → `chunk.meta.low_ocr_confidence = true`.
- **Never** filter low-confidence text out. Surface it; let the operator decide.

## Provenance — non-negotiable

Every chunk carries `(doc_id, page_number, bbox, char_offset_start, char_offset_end, ocr_confidence, ocr_engine)`. If a transform can't preserve a field, the transform is wrong.

## Idempotency

Upload computes `sha256(file_bytes)`. If a document with that hash exists for the case, return the existing `doc_id` and skip re-ingestion. Re-uploads must be cheap and safe.

## Structured extraction

`DocumentMeta` is Pydantic-validated. Two retries with stricter prompt on validation failure (`tenacity`). After exhaustion → set `document.status = "extraction_unclear"`, log a `WARNING`, continue. Never raise to the API caller.
