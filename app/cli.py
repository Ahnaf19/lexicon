"""CLI entry point for Lexicon ingestion.

Usage:
    uv run python -m app.cli ingest <path> [--case-id <uuid>]

<path> can be a single file (.pdf, .jpg, .jpeg, .png) or a directory
(ingests all matching files sequentially).
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import uuid
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
