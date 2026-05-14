"""Rechunking — converts raw OCR blocks into section + window chunks (PRD §5c).

Two row kinds per section:
  "section" — full section text, embedding=NULL (parent-expansion context only, never ranked)
  "window"  — 512-token SentenceSplitter windows, embedding populated by indexing.py
"""

from __future__ import annotations

import bisect
import re
import uuid
from typing import Any

import tiktoken
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sqlalchemy_models import Chunk

# ---------------------------------------------------------------------------
# Module-level singletons (P6)
# ---------------------------------------------------------------------------

_ENCODER = tiktoken.get_encoding("cl100k_base")

_SPLITTER = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=64,
    tokenizer=_ENCODER.encode,
)

# ---------------------------------------------------------------------------
# Section-detection patterns (compiled once)
# ---------------------------------------------------------------------------

_RE_MARKDOWN = re.compile(r"^#{1,6}\s", re.MULTILINE)
_RE_ALLCAPS = re.compile(r"^[A-Z][A-Z\s]{4,}$", re.MULTILINE)
_RE_ARTICLE = re.compile(r"^(ARTICLE|SECTION)\s+[IVX0-9]+", re.MULTILINE)
_RE_NUMBERED = re.compile(r"^\d+(\.\d+)*\s+[A-Z]", re.MULTILINE)

_SECTION_PATTERNS: list[re.Pattern[str]] = [
    _RE_MARKDOWN,
    _RE_ALLCAPS,
    _RE_ARTICLE,
    _RE_NUMBERED,
]


def _find_section_boundaries(text: str) -> list[int]:
    """Return sorted list of character offsets where new sections start.

    Always includes 0 so the entire text is covered.
    """
    hits: set[int] = {0}
    for pat in _SECTION_PATTERNS:
        for m in pat.finditer(text):
            hits.add(m.start())
    return sorted(hits)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _build_cumulative_offsets(source_texts: list[str]) -> list[int]:
    """Cumulative character end-offsets for each source block (P5).

    cumulative_offsets[i] = sum of len(source_texts[0..i]) including separating newlines.
    bisect_left(cumulative_offsets, char_pos) → source block index containing char_pos.
    """
    offsets: list[int] = []
    running = 0
    for t in source_texts:
        running += len(t) + 1  # +1 for the "\n" separator
        offsets.append(running)
    return offsets


def _source_blocks_in_range(
    start: int,
    end: int,
    cumulative_offsets: list[int],
    source_metas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the source-block metas covering [start, end) using bisect (P5)."""
    lo = bisect.bisect_left(cumulative_offsets, start)
    hi = bisect.bisect_left(cumulative_offsets, end)
    hi = min(hi, len(source_metas) - 1)
    return source_metas[lo : hi + 1]


def _aggregate_meta(
    source_blocks: list[dict[str, Any]],
    source_page_nums: list[int],
    cumulative_offsets: list[int],
    start: int,
    end: int,
) -> dict[str, Any]:
    """Build the per-chunk meta dict from the constituent source blocks."""
    lo = bisect.bisect_left(cumulative_offsets, start)
    hi = bisect.bisect_left(cumulative_offsets, end)
    hi = min(hi, len(source_blocks) - 1)

    pages_seen: dict[int, list[dict[str, Any]]] = {}
    confidences: list[float] = []
    has_handwriting = False

    for idx in range(lo, hi + 1):
        m = source_blocks[idx]
        pg = source_page_nums[idx]
        bbox = m.get("bbox", {})
        pages_seen.setdefault(pg, []).append(bbox)
        conf = m.get("ocr_confidence", 1.0)
        confidences.append(conf)
        if m.get("is_handwriting", False):
            has_handwriting = True

    pages_spanned = sorted(pages_seen.keys())
    mean_conf = sum(confidences) / len(confidences) if confidences else 1.0

    return {
        "pages_spanned": pages_spanned,
        "bboxes_per_page": {str(pg): bboxes for pg, bboxes in pages_seen.items()},
        "mean_ocr_confidence": round(mean_conf, 4),
        "has_handwriting": has_handwriting,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def rechunk_document(
    doc_id: uuid.UUID,
    session: AsyncSession,
) -> tuple[int, int]:
    """Replace raw OCR chunks for doc_id with section + window chunks.

    Idempotent: DELETE all existing chunks then bulk INSERT new rows.
    Returns (window_count, section_count).
    """
    # 1. Load source blocks sorted by reading order
    result = await session.execute(
        select(Chunk)
        .where(Chunk.doc_id == doc_id)
        .order_by(Chunk.page_number, Chunk.char_offset_start)
    )
    source_chunks: list[Chunk] = list(result.scalars().all())

    if not source_chunks:
        logger.bind(doc_id=doc_id).warning("rechunk_no_source_blocks")
        return 0, 0

    source_texts = [c.text for c in source_chunks]
    source_metas = [c.meta or {} for c in source_chunks]
    source_pages = [c.page_number for c in source_chunks]

    # 2. Concatenate and build offset index (P5)
    full_text = "\n".join(source_texts)
    cumulative_offsets = _build_cumulative_offsets(source_texts)

    # 3. Find section boundaries
    boundaries = _find_section_boundaries(full_text)
    # Append sentinel
    boundaries.append(len(full_text))

    # 4. Build row groups
    section_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for i in range(len(boundaries) - 1):
        sec_start = boundaries[i]
        sec_end = boundaries[i + 1]
        section_text = full_text[sec_start:sec_end].strip()
        if not section_text:
            continue

        section_id = uuid.uuid4()

        # Determine heading: first line of the section text
        first_line = section_text.split("\n", 1)[0].strip()
        heading = first_line if len(first_line) <= 200 else first_line[:200]

        # Start page = first source block overlapping this section
        lo = bisect.bisect_left(cumulative_offsets, sec_start)
        lo = min(lo, len(source_pages) - 1)
        start_page = source_pages[lo]

        sec_meta = _aggregate_meta(
            source_metas, source_pages, cumulative_offsets, sec_start, sec_end
        )
        sec_meta["kind"] = "section"

        section_rows.append(
            {
                "id": section_id,
                "doc_id": doc_id,
                "page_number": start_page,
                "section_heading": heading,
                "char_offset_start": sec_start,
                "char_offset_end": sec_end,
                "text": section_text,
                "embedding": None,
                "parent_section_id": None,
                "meta": sec_meta,
            }
        )

        # 5. Window chunks via LlamaIndex SentenceSplitter
        windows = _SPLITTER.split_text(section_text)
        # Walk through windows tracking offsets within the section text
        search_start = 0
        for win_text in windows:
            win_text = win_text.strip()
            if not win_text:
                continue

            # Locate this window within section_text (approximate; splitter may
            # normalise whitespace, so we search for the first ~40 chars)
            needle = win_text[:40]
            found = section_text.find(needle, search_start)
            if found == -1:
                found = search_start
            win_start_in_doc = sec_start + found
            win_end_in_doc = win_start_in_doc + len(win_text)
            search_start = max(search_start, found + 1)

            win_meta = _aggregate_meta(
                source_metas,
                source_pages,
                cumulative_offsets,
                win_start_in_doc,
                win_end_in_doc,
            )
            win_meta["kind"] = "window"

            win_lo = bisect.bisect_left(cumulative_offsets, win_start_in_doc)
            win_lo = min(win_lo, len(source_pages) - 1)
            win_page = source_pages[win_lo]

            window_rows.append(
                {
                    "id": uuid.uuid4(),
                    "doc_id": doc_id,
                    "page_number": win_page,
                    "section_heading": heading,
                    "char_offset_start": win_start_in_doc,
                    "char_offset_end": win_end_in_doc,
                    "text": win_text,
                    "embedding": None,
                    "parent_section_id": section_id,
                    "meta": win_meta,
                }
            )

    # 6. Transactional replacement (P4): DELETE then bulk INSERT
    await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))

    all_rows = section_rows + window_rows
    if all_rows:
        await session.execute(insert(Chunk).values(all_rows))

    await session.commit()

    return len(window_rows), len(section_rows)
