"""Deskew scanned PDFs using OpenCV before passing to Marker.

Only applied when the PDF has no native text layer (detected via pdftext).
Returns original bytes unchanged for native PDFs or pages with rotation < 0.5°.
"""

from __future__ import annotations

import io

import cv2
import img2pdf
import numpy as np
from loguru import logger
from pdf2image import convert_from_bytes


def _has_native_text(pdf_bytes: bytes) -> bool:
    """Return True if the PDF has a native text layer extractable by pdftext."""
    import tempfile

    try:
        from pdftext.extraction import plain_text_output

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            text = plain_text_output(tmp.name, sort=False)
        return len(text.strip()) > 50
    except Exception:
        return False


def _deskew_image(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (corrected_image, angle). angle == 0.0 means no rotation applied."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return img_bgr, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0); normalise to [-45, 45)
    if angle < -45:
        angle += 90
    if abs(angle) < 0.5:
        return img_bgr, 0.0
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def deskew_pdf_bytes(pdf_bytes: bytes) -> bytes:
    """Deskew a scanned PDF; returns original bytes for native-text PDFs or near-zero rotation."""
    if _has_native_text(pdf_bytes):
        return pdf_bytes

    try:
        pil_pages = convert_from_bytes(pdf_bytes, dpi=150, fmt="RGB")
        jpeg_pages: list[bytes] = []
        any_rotated = False

        for pil_img in pil_pages:
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            corrected, angle = _deskew_image(img_bgr)
            if angle != 0.0:
                any_rotated = True
                logger.bind(angle=round(angle, 2)).debug("deskew_page_rotated")
            img_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            from PIL import Image as _PilImage

            # Embed DPI so img2pdf computes the correct page size (not 72-DPI default).
            _PilImage.fromarray(img_rgb).save(buf, format="JPEG", quality=95, dpi=(150, 150))
            jpeg_pages.append(buf.getvalue())

        if not any_rotated:
            return pdf_bytes

        result = img2pdf.convert(jpeg_pages)
        logger.bind(pages=len(jpeg_pages)).info("deskew_applied")
        return result if isinstance(result, bytes) else pdf_bytes

    except Exception as exc:
        logger.bind(error=str(exc)).warning("deskew_failed_fallback")
        return pdf_bytes
