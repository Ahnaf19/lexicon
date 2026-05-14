"""Learning loop evaluation — demonstrates the 25-point differentiator (PRD §10).

Procedure:
  Run 1: generate → apply rename script → finalize (corroboration = 1, not promoted)
  Run 2: generate → apply same rename script → finalize (corroboration = 2)
  Run 3: generate → apply same rename script → finalize (corroboration = 3 → PROMOTED)
  Run 4: generate with patterns active — should show lower edit distance

Metrics per run:
  mean_edit_distance_per_item   Levenshtein on (title + description + rationale)
  touch_free_rate               % items with zero edit_events on this checklist
  pattern_application_rate      % items with non-empty learned_from_pattern_ids

Output: eval/results_loop.md and stdout.
Do NOT fudge: if patterns don't promote, log the reason and surface the actual metric.

Config: RETRIEVE_K env var controls top-k retrieval (default 8; eval sets 5 for token savings).
        EVAL_CASE_ID must point to a case that has indexed documents.
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
from sqlalchemy import func, select, text

from app.api.checklist_service import add_item as _svc_add
from app.api.checklist_service import apply_item_patch as _svc_patch
from app.core.db import SessionLocal
from app.core.logging import configure_logging
from app.generation.graph import get_compiled_graph
from app.generation.state import ChecklistState
from app.generation.templates.commercial_contract import COMMERCIAL_CONTRACT
from app.generation.templates.types import ChecklistTemplate
from app.learning.pattern_extractor import extract_patterns
from app.models.pydantic_models import Checklist, ChecklistItem, PartialChecklistItem
from app.models.sqlalchemy_models import Checklist as ChecklistORM
from app.models.sqlalchemy_models import ChecklistItem as ChecklistItemORM
from app.models.sqlalchemy_models import EditEvent as EditEventORM
from app.models.sqlalchemy_models import LearnedPattern as LearnedPatternORM

_EVAL_DIR = Path(__file__).parent
_RESULTS_FILE = _EVAL_DIR / "results_loop.md"
_RENAME_SCRIPT = Path("samples/edit_scripts/rename_counterparty_to_borrower.json")


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


async def apply_edit_script(
    checklist_id: uuid.UUID,
    script_path: Path,
    template: ChecklistTemplate,
) -> int:
    """Load a JSON edit script, resolve slugs to item UUIDs, apply ops in-process.

    Returns the count of successfully applied operations.
    Raises RuntimeError if the DB event count after applying is less than successes
    (the post-condition guard — catches silent persistence failures).
    """
    ops = json.loads(script_path.read_text())

    async with SessionLocal() as session:
        # Baseline event count before any ops.
        start_count_row = await session.execute(
            select(func.count()).where(EditEventORM.checklist_id == checklist_id)
        )
        start_count: int = start_count_row.scalar() or 0

        successes = 0
        for op in ops:
            # Skip comment-only entries.
            if "op" not in op:
                continue
            op_type: str = op["op"]
            slug: str | None = op.get("target_template_item_slug")
            body: dict = op.get("body", {})

            item_id: uuid.UUID | None = None
            if slug is not None:
                # Resolve slug → source_template_item_id via deterministic UUID (no DB join needed).
                tmpl_item = next((i for i in template.items if i.slug == slug), None)
                if tmpl_item is None:
                    print(f"  WARNING: slug '{slug}' not found in template '{template.slug}' — skipping op")
                    continue
                source_uuid = tmpl_item.stable_uuid(template.id)

                row = await session.execute(
                    select(ChecklistItemORM.id).where(
                        ChecklistItemORM.checklist_id == checklist_id,
                        ChecklistItemORM.source_template_item_id == source_uuid,
                    )
                )
                item_id = row.scalar_one_or_none()
                if item_id is None:
                    print(f"  WARNING: no checklist_item found for slug '{slug}' (source_uuid={source_uuid}) — skipping op")
                    continue

            try:
                if op_type == "patch":
                    if item_id is None:
                        print(f"  WARNING: patch op requires target_template_item_slug — skipping")
                        continue
                    partial = PartialChecklistItem(**body)
                    await _svc_patch(session, checklist_id, item_id, partial, actor="eval-script")
                    await session.commit()
                    successes += 1

                elif op_type == "add":
                    # Server assigns a fresh UUID; ignore any id in the script body.
                    body_clean = {k: v for k, v in body.items() if k != "id"}
                    item = ChecklistItem(
                        id=uuid.uuid4(),  # placeholder — service overwrites with new_id
                        source_template_item_id=None,
                        **body_clean,
                    )
                    await _svc_add(session, checklist_id, item, actor="eval-script")
                    await session.commit()
                    successes += 1

                else:
                    print(f"  WARNING: unknown op type '{op_type}' — skipping")
                    continue

            except Exception as exc:
                print(f"  WARNING: op '{op_type}' slug='{slug}' failed: {exc}")
                await session.rollback()

        # Hard post-condition: DB must reflect all claimed successes.
        end_count_row = await session.execute(
            select(func.count()).where(EditEventORM.checklist_id == checklist_id)
        )
        end_count: int = end_count_row.scalar() or 0
        delta = end_count - start_count
        if delta < successes:
            raise RuntimeError(
                f"Post-condition failed: applied {successes} ops but only {delta} "
                f"new edit_events in DB (checklist_id={checklist_id})"
            )

    return successes


async def _get_edit_counts(checklist_id: uuid.UUID) -> dict[uuid.UUID, int]:
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


async def _promoted_pattern_count() -> int:
    async with SessionLocal() as session:
        row = await session.execute(
            select(func.count()).where(LearnedPatternORM.promoted.is_(True))
        )
        return row.scalar() or 0


async def _total_pattern_count() -> int:
    async with SessionLocal() as session:
        row = await session.execute(select(func.count()).select_from(LearnedPatternORM))
        return row.scalar() or 0


async def _compute_metrics(
    run: int,
    checklist_id: uuid.UUID,
    draft_snapshot: dict[uuid.UUID, str],
    edits_applied: int,
) -> dict[str, Any]:
    final_items = await _get_final_items(checklist_id)
    edit_counts = await _get_edit_counts(checklist_id)

    distances = []
    for final_orm in final_items:
        draft_text = draft_snapshot.get(final_orm.id, "")
        final_text = _item_text(final_orm)
        distances.append(Levenshtein.distance(draft_text, final_text))

    mean_edit_dist = sum(distances) / len(distances) if distances else 0.0
    touch_free = sum(1 for it in final_items if edit_counts.get(it.id, 0) == 0)
    touch_free_rate = touch_free / len(final_items) if final_items else 0.0
    pattern_app = sum(1 for it in final_items if (it.learned_from_pattern_ids or []))
    pattern_app_rate = pattern_app / len(final_items) if final_items else 0.0
    promoted = await _promoted_pattern_count()

    return {
        "run": run,
        "checklist_id": str(checklist_id),
        "mean_edit_distance": round(mean_edit_dist, 2),
        "touch_free_rate": round(touch_free_rate, 3),
        "pattern_application_rate": round(pattern_app_rate, 3),
        "promoted_patterns": promoted,
        "item_count": len(final_items),
        "edits_applied": edits_applied,
    }


async def run_eval(case_id: uuid.UUID) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    template = COMMERCIAL_CONTRACT
    run4_metrics: dict[str, Any] | None = None

    for run in range(1, 5):
        print(f"\n=== Run {run} ===")

        checklist = await _generate_checklist(case_id)
        if checklist is None:
            print(f"  ERROR: generation failed on run {run}")
            results.append({"run": run, "error": "generation failed"})
            continue

        checklist_id = checklist.id
        draft_items_raw = await _get_final_items(checklist_id)
        draft_snapshot = {it.id: _item_text(it) for it in draft_items_raw}

        edits_applied = 0
        if run < 4:
            # Apply same rename script on runs 1-3 so corroboration grows to 3 → promoted.
            edits_applied = await apply_edit_script(checklist_id, _RENAME_SCRIPT, template)
            print(f"  edits applied: {edits_applied}")

            if edits_applied == 0:
                raise RuntimeError(
                    "Zero edits applied — eval misconfigured (script/template mismatch). "
                    f"Script: {_RENAME_SCRIPT}, template: {template.slug}"
                )

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

        metrics = await _compute_metrics(run, checklist_id, draft_snapshot, edits_applied)
        print(
            f"  mean_edit_distance={metrics['mean_edit_distance']}"
            f"  touch_free={metrics['touch_free_rate']:.1%}"
            f"  pattern_application={metrics['pattern_application_rate']:.1%}"
            f"  promoted_patterns={metrics['promoted_patterns']}"
        )
        results.append(metrics)

        if run == 4:
            run4_metrics = metrics

    # Categorized failure diagnostics.
    total_patterns = await _total_pattern_count()
    promoted_patterns = await _promoted_pattern_count()

    if total_patterns == 0:
        print("\nWARNING: Edits applied but no patterns extracted — pattern_extractor broken")
    elif promoted_patterns == 0:
        print("\nWARNING: Patterns extracted but none promoted — corroboration/confidence math broken")
    elif run4_metrics is not None and run4_metrics.get("pattern_application_rate", 0.0) == 0.0:
        print("\nWARNING: Patterns promoted but run 4 unchanged — critique/load_template not firing")

    return results


def _write_markdown(results: list[dict[str, Any]], retrieve_k: int) -> None:
    lines = [
        "# Learning Loop Evaluation Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Config: RETRIEVE_K={retrieve_k} (top-k retrieval per item; default 8)",
        "",
        "| Run | edits_applied | mean_edit_distance | touch_free_rate | pattern_application_rate | promoted_patterns |",
        "|-----|--------------|-------------------|-----------------|--------------------------|-------------------|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['run']} | — | ERROR | ERROR | ERROR | — |")
        else:
            lines.append(
                f"| {r['run']} | {r['edits_applied']} "
                f"| {r['mean_edit_distance']} "
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

    # Reduce retrieval top-k to save tokens; annotated in results_loop.md.
    retrieve_k = int(os.environ.setdefault("RETRIEVE_K", "5"))

    case_id_str = os.environ.get("EVAL_CASE_ID")
    if case_id_str:
        case_id = uuid.UUID(case_id_str)
    else:
        case_id = uuid.UUID(int=0)
        print(f"EVAL_CASE_ID not set; using sentinel case_id={case_id}")

    results = await run_eval(case_id)
    _write_markdown(results, retrieve_k)


if __name__ == "__main__":
    asyncio.run(main())
