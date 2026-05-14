"""TrOCR handwriting fallback (microsoft/trocr-large-handwritten).

Lazy-loaded on first call — model load is ~1 GB and ~10 s cold start.
Call re_ocr_block only when Marker flags is_handwriting=True or confidence < threshold.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

_processor: TrOCRProcessor | None = None
_model: VisionEncoderDecoderModel | None = None

_TROCR_MODEL = "microsoft/trocr-large-handwritten"

_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)


def _ensure_loaded() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    global _processor, _model
    if _processor is None or _model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        logger.info("trocr_loading_models")
        _processor = TrOCRProcessor.from_pretrained(_TROCR_MODEL)
        _model = VisionEncoderDecoderModel.from_pretrained(_TROCR_MODEL).to(_DEVICE)
        _model.eval()  # type: ignore[union-attr]
        logger.bind(device=_DEVICE).info("trocr_ready")
    return _processor, _model  # type: ignore[return-value]


def _re_ocr_batch_sync(images: list[Image]) -> list[tuple[str, float]]:
    if not images:
        return []

    processor, model = _ensure_loaded()

    pixel_values = processor(
        images=[img.convert("RGB") for img in images], return_tensors="pt"
    ).pixel_values.to(_DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=128,
        )

    texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    eos_id: int = processor.tokenizer.eos_token_id  # type: ignore[assignment]

    results: list[tuple[str, float]] = []
    for i in range(len(images)):
        token_ids: list[int] = outputs.sequences[i].tolist()
        generated_ids = token_ids[1:]  # strip decoder start token
        try:
            eos_pos = generated_ids.index(eos_id)
            generated_ids = generated_ids[:eos_pos]
        except ValueError:
            pass

        probs: list[float] = []
        for step in range(len(generated_ids)):
            if step >= len(outputs.scores):
                break
            chosen = generated_ids[step]
            token_prob = float(
                torch.softmax(outputs.scores[step][i], dim=-1)[chosen].item()
            )
            probs.append(token_prob)

        confidence = (
            float(math.exp(np.mean(np.log(np.array(probs) + 1e-9)))) if probs else 0.0
        )
        results.append((texts[i].strip(), confidence))

    logger.bind(batch_size=len(images)).debug("trocr_batch_run")
    return results


async def re_ocr_blocks(images: list[Image]) -> list[tuple[str, float]]:
    """Batch re-OCR a list of PIL crops; returns [(text, confidence)] per image."""
    return await asyncio.to_thread(_re_ocr_batch_sync, images)


async def re_ocr_block(image_crop: Image) -> tuple[str, float]:
    """Re-OCR a single PIL crop — thin wrapper over the batched API."""
    results = await re_ocr_blocks([image_crop])
    return results[0]
