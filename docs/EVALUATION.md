# Evaluation

This document covers methodology, results, and interpretation of the Lexicon learning-loop evaluation.

---

## Setup

- **Documents:** 8 total — 5 CUAD contracts (clean PDFs), 2 degraded scans, 1 handwritten exhibit photo
- **Template:** `commercial_contract` (12 items)
- **Retrieval K:** 5 (top-5 per item; default is 8, reduced for faster iteration)
- **LLM:** Groq `llama-3.3-70b-versatile`
- **Runs:** 4 sequential generations on the same case; one edit applied per run

The evaluation script is at `eval/loop_eval.py`. Run it with `uv run python -m eval.run`.

> [!NOTE]
> The eval harness resets learning state (clears `EditEvent` and `LearnedPattern` rows) at the start of each session so that multiple eval runs don't compound. Each session is a clean 4-run loop.

---

## Metrics

| Metric | Definition |
|---|---|
| `edits_applied` | Number of operator edits made to the checklist in this run |
| `mean_edit_distance` | Average number of field changes per item across the 12-item checklist |
| `touch_free_rate` | Share of items requiring zero operator edits (higher = better) |
| `pattern_application_rate` | Share of items whose `learned_from_pattern_ids` is non-empty |
| `promoted_patterns` | Cumulative count of patterns that have reached the corroboration threshold |

---

## Results

### By area

| Area | Metric | Result |
|---|---|---|
| Document processing | 8 docs ingested | Marker at 0.95+ confidence on clean PDFs; TrOCR fallback on handwritten exhibit |
| Grounded retrieval | 271 searchable windows | Avg 2.2–2.3 citations per `present` item; handwritten exhibit findable via semantic embedding despite OCR noise |
| Draft quality | 12-item checklist | 10–11 of 12 items `present` per run; `present ∧ evidence=∅` invariant: 0 rows |
| Improvement loop | 4 runs | 1 pattern promoted at corroboration=3; applied in Run 4; mean_edit_distance 1.58 → 0.00 |

### Loop closure

| Run | edits_applied | mean_edit_distance | touch_free_rate | pattern_application_rate | promoted_patterns |
|---|---|---|---|---|---|
| 1 | 1 | 1.58 | 91.7% | 0.0% | 0 |
| 2 | 1 | 1.58 | 91.7% | 0.0% | 0 |
| 3 | 1 | 1.58 | 91.7% | 0.0% | 1 |
| 4 | 0 | 0.00 | 100.0% | 8.3% | 1 |

---

## Interpretation

**Why do Runs 1–3 have the same metrics?**

Each run applies the same single operator edit (a title rename). The edit accumulates corroboration across runs. By Run 3's finalize step, corroboration crosses 3 — the pattern promotes. But Run 3's generation had *already completed* before the finalize call, so `pattern_application_rate` stays 0% for Run 3.

**Why does Run 4 break the pattern?**

Run 4 starts with the promoted pattern already active. The `critique` node applies the rename rule at draft time — no operator edit is needed. `edits_applied` drops to 0, `mean_edit_distance` to 0.00, `touch_free_rate` to 100%, and `pattern_application_rate` reaches 8.3% (1 of 12 items was modified by a pattern). The loop closed.

**Is 8.3% pattern application meaningful?**

Yes. Only the item that was previously being renamed had a matching pattern. The other 11 items were already correct on first draft. A pattern application rate of 8.3% on a 12-item checklist with a single promoted pattern is the expected result — not a sign of low coverage.

**What would higher edit volume produce?**

With 10–20 operator edits per run rather than 1, multiple patterns would promote within 1–2 runs instead of 3. The corroboration gate is intentionally conservative for a demo with limited edit volume; in production with real operator traffic, the loop would close faster and cover more items.

---

## Raw results

Full JSON at `eval/results_loop.md`.

```json
[
  {"run": 1, "checklist_id": "8badcc7f-...", "mean_edit_distance": 1.58, "touch_free_rate": 0.917, "pattern_application_rate": 0.0, "promoted_patterns": 0, "item_count": 12, "edits_applied": 1},
  {"run": 2, "checklist_id": "b21d86b8-...", "mean_edit_distance": 1.58, "touch_free_rate": 0.917, "pattern_application_rate": 0.0, "promoted_patterns": 0, "item_count": 12, "edits_applied": 1},
  {"run": 3, "checklist_id": "dc7e8e5c-...", "mean_edit_distance": 1.58, "touch_free_rate": 0.917, "pattern_application_rate": 0.0, "promoted_patterns": 1, "item_count": 12, "edits_applied": 1},
  {"run": 4, "checklist_id": "f612138f-...", "mean_edit_distance": 0.0,  "touch_free_rate": 1.0,   "pattern_application_rate": 0.083, "promoted_patterns": 1, "item_count": 12, "edits_applied": 0}
]
```
