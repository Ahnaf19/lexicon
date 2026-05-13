"""Simulate poor-scan degradation on a PDF for OCR robustness testing.

Requires poppler-utils system package:
  macOS:  brew install poppler
  Linux:  apt install poppler-utils

Pipeline applied per page, in order:
  1. Render page at 200 DPI via pdf2image / poppler
  2. Gaussian blur (radius=1.5)
  3. Random rotation in [-3, +3] degrees
  4. Salt-and-pepper noise at 2% pixel density
  5. Contrast reduction to 60% of original
  6. JPEG re-encode at quality=30

Deterministic: numpy.random.seed(42) is set at module load.
CLI: uv run python scripts/degrade.py <input.pdf> <output.pdf>
"""

import io
import sys

import img2pdf
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter

np.random.seed(42)


def degrade(input_path: str, output_path: str) -> None:
    pages = convert_from_path(input_path, dpi=200)
    jpeg_pages: list[bytes] = []

    for img in pages:
        # 1. Gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

        # 2. Random rotation ±3°
        angle = float(np.random.uniform(-3.0, 3.0))
        img = img.rotate(angle, fillcolor="white", expand=False)

        # 3. Salt-and-pepper noise at 2% density (1% salt + 1% pepper)
        arr = np.array(img)
        mask = np.random.random(arr.shape[:2])
        arr[mask < 0.01] = 0
        arr[(mask >= 0.01) & (mask < 0.02)] = 255
        img = Image.fromarray(arr)

        # 4. Contrast reduction to 60%
        img = ImageEnhance.Contrast(img).enhance(0.6)

        # 5. JPEG re-encode at quality=30
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=30)
        jpeg_pages.append(buf.getvalue())

    with open(output_path, "wb") as fh:
        fh.write(img2pdf.convert(jpeg_pages))


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: uv run python scripts/degrade.py <input.pdf> <output.pdf>")
        sys.exit(1)
    degrade(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
