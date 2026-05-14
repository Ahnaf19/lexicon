"""Internal Pydantic types for the ingestion pipeline — not part of the public API schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.pydantic_models import BBox


class Block(BaseModel):
    """One OCR'd text region from Marker (or TrOCR-updated) representing a future chunk."""

    text: str
    page: int
    bbox: BBox
    char_offset_start: int
    char_offset_end: int
    confidence: float = Field(ge=0.0, le=1.0)
    ocr_engine: Literal["marker", "trocr"] = "marker"
    is_handwriting: bool = False
