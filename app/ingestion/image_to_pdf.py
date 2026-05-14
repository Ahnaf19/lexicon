"""Wrap a JPG/PNG image as a single-page PDF so it flows through the same Marker pipeline as native PDFs."""

from __future__ import annotations

import img2pdf


def wrap_image_as_pdf(image_bytes: bytes, mime: str) -> bytes:  # noqa: ARG001
    """Losslessly wrap image bytes as a single-page PDF, preserving native resolution.

    img2pdf autodetects JPEG/PNG from the magic bytes and raises ValueError on
    unsupported formats — let that propagate to the orchestrator's error envelope.
    """
    result: bytes = img2pdf.convert(image_bytes)
    return result
