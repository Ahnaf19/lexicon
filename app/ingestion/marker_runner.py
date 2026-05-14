"""Thin async wrapper around marker-pdf's PdfConverter.

Lazy-loads the Marker/Surya models on first call (heavy, ~3–8 GB VRAM or CPU RAM).
Returns typed Block objects + a MarkerOutput handle that supports image-crop extraction
for the TrOCR fallback.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.schema import BlockTypes

from app.ingestion.models import Block
from app.models.pydantic_models import BBox

_converter: PdfConverter | None = None

# Block types we skip when building the block list — sub-structural units
_SKIP_TYPES = frozenset(
    [BlockTypes.Line, BlockTypes.Span, BlockTypes.Char, BlockTypes.Page]
)


def _get_converter() -> PdfConverter:
    global _converter
    if _converter is None:
        logger.info("marker_loading_models")
        _converter = PdfConverter(artifact_dict=create_model_dict())
        logger.info("marker_ready")
    return _converter


@dataclass
class MarkerOutput:
    """Result of a Marker run — typed blocks + handles for image extraction."""

    blocks: list[Block]
    _marker_blocks: list[Any] = field(repr=False)
    _document: Any = field(repr=False)

    def get_block_image(self, idx: int) -> Any:
        """Return the PIL crop for blocks[idx]; used by TrOCR.

        Marker's highres image is 192 DPI and the rescaling is handled internally.
        """
        return self._marker_blocks[idx].get_image(self._document, highres=True)


def _run_marker_sync(pdf_path: Path) -> MarkerOutput:
    converter = _get_converter()
    # build_document runs layout, OCR, and all processors — fully processed.
    document = converter.build_document(str(pdf_path))

    blocks: list[Block] = []
    marker_blocks: list[Any] = []
    char_offset = 0

    for page in document.pages:
        if page.structure is None:
            continue
        # page.page_id is 0-indexed in Marker; store as 1-indexed for user-facing display
        page_num = page.page_id + 1

        for block_id in page.structure:
            marker_block = page.get_block(block_id)
            if marker_block is None or marker_block.removed:
                continue
            if marker_block.block_type in _SKIP_TYPES:
                continue

            text = marker_block.raw_text(document).strip()
            if not text:
                continue

            raw_bbox = marker_block.polygon.bbox  # [x0, y0, x1, y1] in page-point coords

            # top_k is Dict[BlockTypes, float] when Surya classified the block
            if marker_block.top_k is not None:
                confidence = float(marker_block.top_k.get(marker_block.block_type, 1.0))
            else:
                # Native pdftext extraction — no OCR uncertainty
                confidence = 1.0

            is_handwriting = marker_block.block_type == BlockTypes.Handwriting

            our_block = Block(
                text=text,
                page=page_num,
                bbox=BBox(x0=raw_bbox[0], y0=raw_bbox[1], x1=raw_bbox[2], y1=raw_bbox[3]),
                char_offset_start=char_offset,
                char_offset_end=char_offset + len(text),
                confidence=confidence,
                ocr_engine="marker",
                is_handwriting=is_handwriting,
            )
            blocks.append(our_block)
            marker_blocks.append(marker_block)
            char_offset += len(text) + 1

    low_conf_count = sum(1 for b in blocks if b.confidence < 0.5)
    mean_conf = sum(b.confidence for b in blocks) / len(blocks) if blocks else 0.0
    logger.bind(
        page_count=len(document.pages),
        block_count=len(blocks),
        mean_confidence=round(mean_conf, 3),
        low_conf_count=low_conf_count,
    ).info("marker_done")

    return MarkerOutput(blocks=blocks, _marker_blocks=marker_blocks, _document=document)


async def run_marker(pdf_path: Path) -> MarkerOutput:
    """Run Marker OCR asynchronously (offloads the CPU-heavy work to a thread)."""
    return await asyncio.to_thread(_run_marker_sync, pdf_path)
