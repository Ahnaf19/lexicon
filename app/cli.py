"""CLI entry point for Lexicon ingestion, indexing, and search.

Usage:
    uv run python -m app.cli ingest <path> [--case-id <uuid>]
    uv run python -m app.cli reindex [doc_id | --case-id X | --all]
    uv run python -m app.cli search "<query>" --case-id X [--k 5]
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
