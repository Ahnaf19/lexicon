# Learning module — edit capture + pattern extraction

PRD §5f. The 25-point differentiator. **Protect from scope cuts.**

## Edit events

Every UI mutation (PATCH/POST/DELETE on checklist items) writes one `edit_events` row. Event types:

`item_added | item_removed | item_renamed | status_changed | evidence_added | evidence_corrected | category_reordered | item_text_rewritten | required_toggled`

Payload schema differs per type — model as a Pydantic discriminated union.

## Few-shot example bank

On `finalize`, every edited item produces one `few_shot_examples` row: `(original_draft, final_item, context_embedding)`. Embedding is over `template_item.title + category + retrieved_evidence_summary`. At `draft_item` time, retrieve top-3 by cosine, filtered by `doc_type`. This is the CIPHER pattern (Gao et al., 2024).

## Learned patterns — promotion rules

- **Created** the first time the extractor identifies it.
- **Promoted** (i.e., applied at draft/critique time) only when `corroborating_edit_count >= 3` and `confidence >= 0.7`.
- Operator dismissal decrements corroboration count; demoted if it falls below 3.

## Application points

1. Few-shot exemplars in `draft_item` prompt (CIPHER, retrieval-driven).
2. Template mutation for promoted `template_addition` / `template_removal`.
3. `critique` node rewrites items violating promoted rules.

## Don't

- Don't apply un-promoted patterns. Single-edit signals are noise.
- Don't fine-tune on edit data. Retrieval-augmented few-shot is the design.
- Don't compute a "side-by-side diff" and call it the improvement loop. The brief explicitly rules this out.
