"""CLI entry point for Lexicon ingestion, indexing, search, and checklist generation.

Usage:
    uv run python -m app.cli ingest <path> [--case-id <uuid>]
    uv run python -m app.cli reindex [doc_id | --case-id X | --all]
    uv run python -m app.cli search "<query>" --case-id X [--k 5]
    uv run python -m app.cli checklist generate --case-id X [--template commercial_contract|nda]
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from app.core.db import SessionLocal
from app.core.logging import configure_logging
from app.ingestion.orchestrator import ingest_document
from app.models.sqlalchemy_models import Document

_DEFAULT_CASE = uuid.UUID(int=0)

app = typer.Typer(help="Lexicon document ingestion CLI")
console = Console()

_SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}
_MIME_MAP = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _resolve_mime(path: Path) -> str:
    return _MIME_MAP.get(path.suffix.lower(), "application/octet-stream")


async def _ingest_and_print(path: Path, case_id: uuid.UUID) -> None:
    mime = _resolve_mime(path)
    file_bytes = path.read_bytes()
    rprint(f"[bold]>>>[/bold] ingesting [cyan]{path.name}[/cyan] ({mime}) ...")

    doc_id = await ingest_document(
        file_bytes=file_bytes,
        filename=path.name,
        mime=mime,
        case_id=case_id,
        session_factory=SessionLocal,
    )

    async with SessionLocal() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            rprint(f"[red]    ERROR:[/red] document {doc_id} not found after ingestion")
            return

        output = {
            "document_id": str(doc.id),
            "filename": doc.original_filename,
            "status": doc.status,
            "doc_type": doc.doc_type,
            "ocr_engine": doc.ocr_engine,
            "total_pages": doc.total_pages,
            "meta": doc.meta,
        }
        rprint(f"[green]    OK[/green] status=[bold]{doc.status}[/bold]")
        print(json.dumps(output, indent=2, default=str))


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest"),
    case_id: uuid.UUID = typer.Option(_DEFAULT_CASE, "--case-id", help="Case UUID"),
) -> None:
    """Ingest one or more documents and print the resulting DocumentMeta."""
    configure_logging()

    if path.is_file():
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            typer.echo(f"Unsupported file type: {path.suffix}", err=True)
            raise typer.Exit(1)
        files = [path]
    elif path.is_dir():
        files = sorted(
            f for f in path.iterdir() if f.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        if not files:
            typer.echo(f"No supported files in {path}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo(f"Path not found: {path}", err=True)
        raise typer.Exit(1)

    async def _run_all() -> None:
        # Pre-load Marker + TrOCR before timing the first document so cold-start
        # doesn't inflate the first file's wall-clock time.
        from app.ingestion.marker_runner import _get_converter
        from app.ingestion.trocr_fallback import _ensure_loaded

        rprint("[dim]   warming models...[/dim]")
        await asyncio.gather(
            asyncio.to_thread(_get_converter),
            asyncio.to_thread(_ensure_loaded),
        )
        rprint("[dim]   models ready[/dim]")

        for file in files:
            await _ingest_and_print(file, case_id)

    asyncio.run(_run_all())


@app.command()
def reindex(
    doc_id: Optional[uuid.UUID] = typer.Argument(None, help="Single document UUID to reindex"),
    case_id: Optional[uuid.UUID] = typer.Option(None, "--case-id", help="Reindex all docs in case"),
    all_docs: bool = typer.Option(False, "--all", help="Reindex every document in the DB"),
) -> None:
    """Rechunk and re-embed documents, making them retrieval-ready."""
    configure_logging()
    from sqlalchemy import select

    from app.retrieval.indexing import index_document

    async def _run() -> None:
        async with SessionLocal() as session:
            stmt = select(Document.id, Document.original_filename)
            if doc_id is not None:
                stmt = stmt.where(Document.id == doc_id)
            elif case_id is not None:
                stmt = stmt.where(Document.case_id == case_id)
            elif all_docs:
                pass  # no filter — all docs
            else:
                rprint("[red]Specify a doc_id, --case-id, or --all[/red]")
                raise typer.Exit(1)

            result = await session.execute(stmt)
            targets = result.all()

        if not targets:
            rprint("[yellow]No matching documents found.[/yellow]")
            return

        rprint(f"[bold]Reindexing {len(targets)} document(s)...[/bold]")
        sem = asyncio.Semaphore(2)

        async def _index_one(did: uuid.UUID, fname: str) -> None:
            async with sem:
                rprint(f"  [cyan]{fname}[/cyan] ({did}) ...")
                try:
                    await index_document(did)
                    rprint(f"  [green]OK[/green] {fname}")
                except Exception as exc:
                    rprint(f"  [red]FAIL[/red] {fname}: {exc}")

        await asyncio.gather(*[_index_one(did, fname) for did, fname in targets])

    asyncio.run(_run())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query string"),
    case_id: uuid.UUID = typer.Option(..., "--case-id", help="Case UUID to search within"),
    k: int = typer.Option(5, "--k", help="Number of results to return"),
) -> None:
    """Hybrid dense + sparse search over indexed documents."""
    configure_logging()
    from sqlalchemy import select

    from app.retrieval.hybrid_search import search as _search

    async def _run() -> None:
        async with SessionLocal() as session:
            hits = await _search(query, case_id, session, k=k)

        if not hits:
            rprint("[yellow]No results found.[/yellow]")
            return

        # Fetch short filenames for display
        async with SessionLocal() as session:
            doc_ids = list({h.doc_id for h in hits})
            name_result = await session.execute(
                select(Document.id, Document.original_filename).where(
                    Document.id.in_(doc_ids)
                )
            )
            name_map = {row[0]: row[1] for row in name_result}

        table = Table(title=f'Search: "{query}"', show_lines=True)
        table.add_column("Rank", style="bold", width=4)
        table.add_column("Doc", style="cyan", max_width=30)
        table.add_column("Page", width=4)
        table.add_column("Dense", width=5)
        table.add_column("Sparse", width=6)
        table.add_column("RRF", width=7)
        table.add_column("Snippet", max_width=60)

        for rank, hit in enumerate(hits, 1):
            fname = name_map.get(hit.doc_id, str(hit.doc_id))[:28]
            table.add_row(
                str(rank),
                fname,
                str(hit.page_number),
                str(hit.dense_rank) if hit.dense_rank is not None else "—",
                str(hit.sparse_rank) if hit.sparse_rank is not None else "—",
                f"{hit.retrieval_score:.4f}",
                hit.snippet[:100],
            )

        console.print(table)

    asyncio.run(_run())


checklist_app = typer.Typer(help="Checklist generation commands")
app.add_typer(checklist_app, name="checklist")


@checklist_app.command("generate")
def checklist_generate(
    case_id: uuid.UUID = typer.Option(..., "--case-id", help="Case UUID"),
    template: str = typer.Option("commercial_contract", "--template", help="Template slug: commercial_contract or nda"),
) -> None:
    """Generate a checklist for a case; streams per-item progress."""
    configure_logging()
    import json as _json

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from app.generation.graph import get_compiled_graph
    from app.generation.state import ChecklistState
    from app.generation.templates import TEMPLATES
    from app.models.pydantic_models import Checklist

    if template not in TEMPLATES:
        rprint(f"[red]Unknown template '{template}'. Available: {list(TEMPLATES)}[/red]")
        raise typer.Exit(1)

    tmpl = TEMPLATES[template]
    slugs = [item.slug for item in tmpl.items]

    async def _run() -> None:
        initial_state: ChecklistState = {
            "case_id": case_id,
            "template_slug": template,
            "document_ids": [],
            "errors": [],
        }
        graph = get_compiled_graph()
        completed: set[str] = set()
        final_state: dict = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            tasks = {slug: progress.add_task(f"[cyan]{slug}[/cyan]", total=1) for slug in slugs}

            # astream yields incremental state updates; the last chunk is the full final state.
            async for chunk in graph.astream(initial_state, stream_mode="updates"):
                final_state.update(chunk)
                # Each chunk is {node_name: node_output_dict}
                for node_name, node_out in chunk.items():
                    if node_name == "validate_item" and isinstance(node_out, dict):
                        in_progress = node_out.get("items_in_progress", {})
                        for s in slugs:
                            if s not in completed:
                                entry = in_progress.get(s)
                                if entry is not None and not isinstance(entry, dict):
                                    completed.add(s)
                                    if s in tasks:
                                        progress.advance(tasks[s])
                                        progress.update(
                                            tasks[s],
                                            description=f"[green]{s} ✓[/green]",
                                        )

        # astream "updates" mode: final_state = {node_name: output_dict, ...}
        assemble_out = final_state.get("assemble") or {}
        checklist_out = assemble_out.get("checklist") if isinstance(assemble_out, dict) else None

        if checklist_out is None:
            rprint("[red]ERROR: No checklist produced. Check logs for errors.[/red]")
            raise typer.Exit(1)

        if isinstance(checklist_out, Checklist):
            output = checklist_out.model_dump(mode="json")
        else:
            output = checklist_out

        console.print_json(_json.dumps(output, default=str))

    asyncio.run(_run())


learning_app = typer.Typer(help="Learning loop — edit capture, pattern extraction, pattern list")
app.add_typer(learning_app, name="learning")


@learning_app.command("edits-simulate")
def learning_edits_simulate(
    checklist_id: uuid.UUID = typer.Option(..., "--checklist-id", help="Target checklist UUID"),
    script: Path = typer.Argument(..., help="Path to edit script JSON file"),
) -> None:
    """Apply a canned edit script to a checklist via HTTP.

    Each script is a JSON array of {method, path, body} objects.
    """
    configure_logging()
    import httpx as _httpx

    if not script.exists():
        rprint(f"[red]Script not found: {script}[/red]")
        raise typer.Exit(1)

    import json as _json

    steps = _json.loads(script.read_text())

    from app.main import app as _fastapi_app

    async def _run() -> None:
        async with _httpx.AsyncClient(
            transport=_httpx.ASGITransport(app=_fastapi_app),  # type: ignore[arg-type]
            base_url="http://testserver",
        ) as client:
            for i, step in enumerate(steps, 1):
                method = step["method"].upper()
                path = step["path"]
                body = step.get("body")
                rprint(f"[dim]  step {i}: {method} {path}[/dim]")
                resp = await client.request(method, path, json=body)
                if resp.status_code >= 400:
                    rprint(f"  [red]ERROR {resp.status_code}:[/red] {resp.text[:200]}")
                else:
                    rprint(f"  [green]{resp.status_code}[/green] {resp.text[:200]}")

    asyncio.run(_run())


@learning_app.command("finalize")
def learning_finalize(
    checklist_id: uuid.UUID = typer.Argument(..., help="Checklist UUID to finalize"),
) -> None:
    """Finalize a checklist and wait for pattern extraction to complete."""
    configure_logging()
    from app.learning.pattern_extractor import extract_patterns

    async def _run() -> None:
        rprint(f"[bold]Finalizing checklist {checklist_id}...[/bold]")
        async with SessionLocal() as session:
            from datetime import datetime, timezone
            from sqlalchemy import update as _update
            from app.models.sqlalchemy_models import Checklist as ChecklistORM

            await session.execute(
                _update(ChecklistORM)
                .where(ChecklistORM.id == checklist_id)
                .values(
                    finalized_at=datetime.now(timezone.utc),
                    status="finalized",
                )
            )
            await session.commit()

        rprint("[dim]  running pattern extraction...[/dim]")
        await extract_patterns(checklist_id)
        rprint("[green]  done[/green]")

    asyncio.run(_run())


@learning_app.command("patterns-list")
def learning_patterns_list(
    promoted_only: bool = typer.Option(False, "--promoted-only", help="Show only promoted patterns"),
    doc_type: Optional[str] = typer.Option(None, "--doc-type", help="Filter by doc_type"),
) -> None:
    """Pretty-print learned_patterns table."""
    configure_logging()
    from sqlalchemy import select

    from app.models.pydantic_models import LearnedPattern
    from app.models.sqlalchemy_models import LearnedPattern as LearnedPatternORM

    async def _run() -> None:
        async with SessionLocal() as session:
            stmt = select(LearnedPatternORM)
            if promoted_only:
                stmt = stmt.where(LearnedPatternORM.promoted.is_(True))
            if doc_type:
                stmt = stmt.where(LearnedPatternORM.doc_type_scope == doc_type)
            stmt = stmt.order_by(LearnedPatternORM.created_at.desc())
            result = await session.execute(stmt)
            rows = list(result.scalars().all())

        if not rows:
            rprint("[yellow]No patterns found.[/yellow]")
            return

        table = Table(title="Learned Patterns", show_lines=True)
        table.add_column("Type", style="cyan", max_width=18)
        table.add_column("Scope", max_width=20)
        table.add_column("Count", width=5)
        table.add_column("Conf", width=5)
        table.add_column("Promoted", width=8)
        table.add_column("Rule", max_width=50)

        for row in rows:
            import json as _json

            table.add_row(
                row.pattern_type,
                row.doc_type_scope,
                str(row.corroborating_edit_count),
                f"{row.confidence:.2f}",
                "[green]YES[/green]" if row.promoted else "[red]no[/red]",
                _json.dumps(row.rule_json, default=str)[:100],
            )
        console.print(table)

    asyncio.run(_run())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
