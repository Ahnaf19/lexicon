"""Learning loop evaluation — demonstrates the 25-point differentiator (PRD §10).

Procedure:
  Run 1: generate → apply edit script → finalize (corroboration = 1, not promoted)
  Run 2: generate → apply same edits → finalize (corroboration = 2)
  Run 3: generate → apply same edits → finalize (corroboration = 3 → PROMOTED)
  Run 4: generate with patterns active — should show lower edit distance

Metrics per run:
  mean_edit_distance_per_item   Levenshtein on (title + description + rationale)
  touch_free_rate               % items with zero edit_events on this checklist
  pattern_application_rate      % items with non-empty learned_from_pattern_ids

Output: eval/results_loop.md and stdout.
Do NOT fudge: if patterns don't promote, log the reason and surface the actual metric.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein
from sqlalchemy import func, select

from app.core.db import SessionLocal
from app.core.logging import configure_logging
from app.generation.graph import get_compiled_graph
from app.generation.state import ChecklistState
from app.learning.pattern_extractor import extract_patterns
from app.models.pydantic_models import Checklist, ChecklistItem
from app.models.sqlalchemy_models import Checklist as ChecklistORM
from app.models.sqlalchemy_models import ChecklistItem as ChecklistItemORM
from app.models.sqlalchemy_models import EditEvent as EditEventORM
from app.models.sqlalchemy_models import LearnedPattern as LearnedPatternORM

_EVAL_DIR = Path(__file__).parent
_RESULTS_FILE = _EVAL_DIR / "results_loop.md"


async def _generate_checklist(case_id: uuid.UUID, template: str = "commercial_contract") -> Checklist | None:
    initial_state: ChecklistState = {
        "case_id": case_id,
        "template_slug": template,
        "document_ids": [],
        "errors": [],
    }
    graph = get_compiled_graph()
    final: dict = {}
    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        final.update(chunk)
    assemble_out = final.get("assemble") or {}
    return assemble_out.get("checklist") if isinstance(assemble_out, dict) else None


async def _apply_counterparty_rename(checklist: Checklist, checklist_id: uuid.UUID) -> int:
    """Find items with 'Counterparty' in title and rename; return edit count."""
    edits = 0
    async with SessionLocal() as session:
        from datetime import datetime, timezone
        from app.learning.edit_capture import write_edit_event
        from app.models.pydantic_models import ItemRenamed
        from app.models.sqlalchemy_models import ChecklistItem as ChecklistItemORM
        from sqlalchemy import update

        for item in checklist.items:
            if "counterparty" in item.title.lower():
                new_title = item.title.replace("Counterparty", "Borrower").replace(
                    "counterparty", "borrower"
                )
                await session.execute(
                    update(ChecklistItemORM)
                    .where(ChecklistItemORM.id == item.id)
                    .values(title=new_title)
                )
                event = ItemRenamed(
                    item_id=item.id, old_title=item.title, new_title=new_title
                )
                await write_edit_event(
                    session=session,
                    checklist_id=checklist_id,
                    item_id=item.id,
                    event=event,
                    actor="eval-script",
                )
                edits += 1
        await session.commit()
    return edits


async def _get_edit_counts(checklist_id: uuid.UUID) -> dict[uuid.UUID, int]:
    """Return {item_id: edit_count} for this checklist."""
    async with SessionLocal() as session:
        result = await session.execute(
            select(EditEventORM.checklist_item_id, func.count().label("cnt"))
            .where(EditEventORM.checklist_id == checklist_id)
            .group_by(EditEventORM.checklist_item_id)
        )
        return {row[0]: row[1] for row in result if row[0] is not None}


async def _get_final_items(checklist_id: uuid.UUID) -> list[ChecklistItemORM]:
    async with SessionLocal() as session:
        result = await session.execute(
            select(ChecklistItemORM).where(ChecklistItemORM.checklist_id == checklist_id)
        )
        return list(result.scalars().all())


def _item_text(item: Any) -> str:
    title = getattr(item, "title", "") or ""
    desc = getattr(item, "description", "") or ""
    rationale = getattr(item, "rationale", "") or ""
    return title + " " + desc + " " + rationale


async def _compute_metrics(
    run: int,
    checklist: Checklist,
    checklist_id: uuid.UUID,
    draft_items: dict[uuid.UUID, str],
) -> dict[str, Any]:
    final_items = await _get_final_items(checklist_id)
    edit_counts = await _get_edit_counts(checklist_id)

    distances = []
    for final_orm in final_items:
        draft_text = draft_items.get(final_orm.id, "")
        final_text = _item_text(final_orm)
        distances.append(Levenshtein.distance(draft_text, final_text))

    mean_edit_dist = sum(distances) / len(distances) if distances else 0.0
    touch_free = sum(1 for it in final_items if edit_counts.get(it.id, 0) == 0)
    touch_free_rate = touch_free / len(final_items) if final_items else 0.0

    pattern_app = sum(
        1 for it in final_items if (it.learned_from_pattern_ids or [])
    )
    pattern_app_rate = pattern_app / len(final_items) if final_items else 0.0

    async with SessionLocal() as session:
        promoted = await session.execute(
            select(func.count()).where(LearnedPatternORM.promoted.is_(True))
        )
        promoted_count = promoted.scalar() or 0

    return {
        "run": run,
        "checklist_id": str(checklist_id),
        "mean_edit_distance": round(mean_edit_dist, 2),
        "touch_free_rate": round(touch_free_rate, 3),
        "pattern_application_rate": round(pattern_app_rate, 3),
        "promoted_patterns": promoted_count,
        "item_count": len(final_items),
        "edits_applied": sum(edit_counts.values()),
    }


async def run_eval(case_id: uuid.UUID) -> list[dict[str, Any]]:
    results = []

    for run in range(1, 5):
        print(f"\n=== Run {run} ===")

        # Generate checklist.
        checklist = await _generate_checklist(case_id)
        if checklist is None:
            print(f"  ERROR: generation failed on run {run}")
            results.append({"run": run, "error": "generation failed"})
            continue

        checklist_id = checklist.id
        # Snapshot draft text (post-validate, pre-edit) from DB.
        draft_items_raw = await _get_final_items(checklist_id)
        draft_snapshot = {it.id: _item_text(it) for it in draft_items_raw}

        # Apply edits on runs 1-3 only (run 4 tests the learned patterns).
        edits_applied = 0
        if run < 4:
            edits_applied = await _apply_counterparty_rename(checklist, checklist_id)
            print(f"  edits applied: {edits_applied}")

            # Finalize + extract patterns.
            async with SessionLocal() as session:
                from sqlalchemy import update as _upd

                await session.execute(
                    _upd(ChecklistORM)
                    .where(ChecklistORM.id == checklist_id)
                    .values(finalized_at=datetime.now(timezone.utc), status="finalized")
                )
                await session.commit()

            await extract_patterns(checklist_id)

        metrics = await _compute_metrics(run, checklist, checklist_id, draft_snapshot)
        print(
            f"  mean_edit_distance={metrics['mean_edit_distance']}"
            f"  touch_free={metrics['touch_free_rate']:.1%}"
            f"  pattern_application={metrics['pattern_application_rate']:.1%}"
            f"  promoted_patterns={metrics['promoted_patterns']}"
        )

        if run == 3 and metrics["promoted_patterns"] == 0:
            print(
                "  WARNING: no patterns promoted after 3 corroborating runs."
                " Check that edit_events contain enough signal."
                " The run-4 metrics will show 0% pattern_application_rate — this is honest."
            )

        results.append(metrics)

    return results


def _write_markdown(results: list[dict[str, Any]]) -> None:
    lines = [
        "# Learning Loop Evaluation Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Run | mean_edit_distance | touch_free_rate | pattern_application_rate | promoted_patterns |",
        "|-----|-------------------|-----------------|--------------------------|-------------------|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['run']} | ERROR | ERROR | ERROR | — |")
        else:
            lines.append(
                f"| {r['run']} | {r['mean_edit_distance']} "
                f"| {r['touch_free_rate']:.1%} "
                f"| {r['pattern_application_rate']:.1%} "
                f"| {r['promoted_patterns']} |"
            )
    lines += [
        "",
        "Expected trend: mean_edit_distance DOWN from run 1→4; "
        "touch_free_rate UP; pattern_application_rate > 0 at run 4.",
        "",
        "Note: metrics are honest — if patterns did not promote, run 4 shows 0% pattern_application_rate.",
        "",
        "Raw JSON:",
        "```json",
        json.dumps(results, indent=2, default=str),
        "```",
    ]
    _RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {_RESULTS_FILE}")


async def main() -> None:
    configure_logging()

    case_id_str = os.environ.get("EVAL_CASE_ID")
    if case_id_str:
        case_id = uuid.UUID(case_id_str)
    else:
        case_id = uuid.UUID(int=0)
        print(f"EVAL_CASE_ID not set; using sentinel case_id={case_id}")

    results = await run_eval(case_id)
    _write_markdown(results)


if __name__ == "__main__":
    asyncio.run(main())
