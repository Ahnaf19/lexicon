# Learning Loop Evaluation Results

Generated: 2026-05-14T22:29:07.051672+00:00
Config: RETRIEVE_K=5 (top-k retrieval per item; default 8)

| Run | edits_applied | mean_edit_distance | touch_free_rate | pattern_application_rate | promoted_patterns |
|-----|--------------|-------------------|-----------------|--------------------------|-------------------|
| 1 | 1 | 1.58 | 91.7% | 0.0% | 0 |
| 2 | 1 | 1.58 | 91.7% | 0.0% | 0 |
| 3 | 1 | 1.58 | 91.7% | 0.0% | 1 |
| 4 | 0 | 0.0 | 100.0% | 8.3% | 1 |

Expected trend: mean_edit_distance DOWN from run 1→4; touch_free_rate UP; pattern_application_rate > 0 at run 4.

Note: metrics are honest — if patterns did not promote, run 4 shows 0% pattern_application_rate.

Raw JSON:
```json
[
  {
    "run": 1,
    "checklist_id": "8badcc7f-67d7-4a72-81fa-ba24934dd891",
    "mean_edit_distance": 1.58,
    "touch_free_rate": 0.917,
    "pattern_application_rate": 0.0,
    "promoted_patterns": 0,
    "item_count": 12,
    "edits_applied": 1
  },
  {
    "run": 2,
    "checklist_id": "b21d86b8-b3c0-45ac-bc3e-117c8f59759f",
    "mean_edit_distance": 1.58,
    "touch_free_rate": 0.917,
    "pattern_application_rate": 0.0,
    "promoted_patterns": 0,
    "item_count": 12,
    "edits_applied": 1
  },
  {
    "run": 3,
    "checklist_id": "dc7e8e5c-e4de-46d2-8a9b-8d0cb35e0a78",
    "mean_edit_distance": 1.58,
    "touch_free_rate": 0.917,
    "pattern_application_rate": 0.0,
    "promoted_patterns": 1,
    "item_count": 12,
    "edits_applied": 1
  },
  {
    "run": 4,
    "checklist_id": "f612138f-38cf-45ac-a070-9271b6350ed3",
    "mean_edit_distance": 0.0,
    "touch_free_rate": 1.0,
    "pattern_application_rate": 0.083,
    "promoted_patterns": 1,
    "item_count": 12,
    "edits_applied": 0
  }
]
```