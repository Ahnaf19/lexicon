"""Structured extraction: regex layer + LLM layer → DocumentMeta.

Two retries on Pydantic validation failure (tenacity). On exhaustion, returns partial
meta with status="extraction_unclear" — never raises to the orchestrator.
"""

from __future__ import annotations

import contextlib
import re
from datetime import date, datetime
from typing import Literal
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, ValidationError
from tenacity import RetryCallState

from app.core.llm import get_chat_model
from app.models.pydantic_models import (
    DefinedTerm,
    DocumentMeta,
    MonetaryAmount,
    Party,
    SignatureBlock,
)

# ---------------------------------------------------------------------------
# Lenient internal extraction schemas — qwen3/Ollama outputs null for optional
# arrays and null for nested required fields; we accept everything as nullable
# here and coerce to canonical types before building DocumentMeta.
# ---------------------------------------------------------------------------


class _PartyOut(BaseModel):
    name: str | None = None
    role: str | None = None


class _MoneyOut(BaseModel):
    amount: float | None = None
    currency: str | None = None
    context: str | None = None


class _TermOut(BaseModel):
    term: str | None = None
    definition: str | None = None
    page: int | str | None = None


class _SigOut(BaseModel):
    signatory: str | None = None
    party: str | None = None
    page: int | str | None = None
    signed: bool | None = None


class _ExtractOut(BaseModel):
    doc_type: Literal[
        "nda",
        "employment",
        "commercial_contract",
        "license",
        "service",
        "distribution",
        "maintenance",
        "strategic_alliance",
        "loan_agreement",
        "other",
    ] = "other"
    parties: list[_PartyOut] | None = None
    effective_date: str | None = None
    monetary_terms: list[_MoneyOut] | None = None
    defined_terms: list[_TermOut] | None = None
    exhibits_referenced: list[str] | None = None
    signature_blocks: list[_SigOut] | None = None
    governing_law: str | None = None
    confidence: float | None = None


def _coerce_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%B %d %Y"):
        try:
            return (
                date.fromisoformat(value)
                if fmt == "%Y-%m-%d"
                else datetime.strptime(value, fmt).date()
            )
        except ValueError:
            continue
    return None


def _coerce_int(value: int | str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _to_canonical(out: _ExtractOut, doc_id: UUID) -> DocumentMeta:
    parties = [Party(name=p.name or "unknown", role=p.role) for p in (out.parties or []) if p.name]
    monetary_terms = [
        MonetaryAmount(amount=m.amount, currency=m.currency or "USD", context=m.context)
        for m in (out.monetary_terms or [])
        if m.amount is not None
    ]
    defined_terms = [
        DefinedTerm(
            term=t.term or "",
            definition=t.definition or "",
            page=_coerce_int(t.page),
        )
        for t in (out.defined_terms or [])
        if t.term
    ]
    signature_blocks = [
        SignatureBlock(
            signatory=s.signatory or "unknown",
            party=s.party,
            page=_coerce_int(s.page) or 0,
            signed=s.signed if s.signed is not None else True,
        )
        for s in (out.signature_blocks or [])
    ]
    return DocumentMeta(
        doc_id=doc_id,
        doc_type=out.doc_type,
        parties=parties,
        effective_date=_coerce_date(out.effective_date),
        monetary_terms=monetary_terms,
        defined_terms=defined_terms,
        exhibits_referenced=out.exhibits_referenced or [],
        signature_blocks=signature_blocks,
        governing_law=out.governing_law,
        confidence=out.confidence if out.confidence is not None else 0.5,
    )


# ---------------------------------------------------------------------------
# Regex layer
# ---------------------------------------------------------------------------

_DATE_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATE_US = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_DATE_WRITTEN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
)
_MONEY = re.compile(r"\$\s*[\d,]+(?:\.\d{2})?")
_EXHIBIT = re.compile(r"\bExhibit\s+[A-Z]\b|\bSchedule\s+\d+\b")
_SIG = re.compile(r"/s/|By:\s*_{2,}", re.IGNORECASE)


def _regex_extract(text: str) -> dict[str, object]:
    """Fast deterministic pre-pass over the full document text."""
    dates = _DATE_ISO.findall(text) + _DATE_US.findall(text) + _DATE_WRITTEN.findall(text)
    money_strs = _MONEY.findall(text)
    exhibits = list(dict.fromkeys(_EXHIBIT.findall(text)))  # deduplicated, order-preserving
    has_sig = bool(_SIG.search(text))

    monetary_terms: list[MonetaryAmount] = []
    for m in money_strs[:10]:  # cap to avoid bloated payloads
        raw = m.replace("$", "").replace(",", "").strip()
        with contextlib.suppress(Exception):
            monetary_terms.append(MonetaryAmount(amount=float(raw), currency="USD"))

    signature_blocks: list[SignatureBlock] = []
    if has_sig:
        signature_blocks.append(
            SignatureBlock(signatory="unknown", party=None, page=0, signed=True)
        )

    return {
        "raw_dates": dates,
        "monetary_terms": monetary_terms,
        "exhibits_referenced": exhibits,
        "signature_blocks": signature_blocks,
    }


# ---------------------------------------------------------------------------
# LLM layer with tenacity retry on ValidationError
# ---------------------------------------------------------------------------

_SYSTEM_BASE = (
    "You are a legal document analyser. Extract a DocumentMeta JSON object from the "
    "provided contract text. Use ONLY facts that appear verbatim in the text. "
    "Choose doc_type from: nda, employment, commercial_contract, license, service, "
    "distribution, maintenance, strategic_alliance, loan_agreement, other. "
    "If uncertain about a field, omit it or use null. "
    "Return a single valid JSON object — no prose, no markdown."
)

_SYSTEM_STRICT = (
    _SYSTEM_BASE + "\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {error}\n"
    "Return ONLY a JSON object matching the DocumentMeta schema exactly."
)


def _build_retry_state_logger(doc_id: UUID) -> object:
    def _before_retry(retry_state: RetryCallState) -> None:
        logger.bind(doc_id=doc_id, attempt=retry_state.attempt_number).warning(
            "structured_extract_retry"
        )

    return _before_retry


async def _llm_extract(text: str, doc_id: UUID, attempt: int, last_error: str) -> DocumentMeta:
    system = _SYSTEM_BASE if attempt == 1 else _SYSTEM_STRICT.format(error=last_error)
    model = get_chat_model(role="fast").with_structured_output(_ExtractOut)
    prompt = f"{system}\n\nDOCUMENT TEXT (truncated to 8000 chars):\n{text[:8000]}"
    result = await model.ainvoke(prompt)
    if not isinstance(result, _ExtractOut):
        raise ValueError(f"Unexpected LLM output type: {type(result)}")
    return _to_canonical(result, doc_id)


async def extract_document_meta(
    full_text: str, doc_id: UUID
) -> tuple[DocumentMeta, Literal["ok", "extraction_unclear"]]:
    """Extract DocumentMeta from OCR'd text.

    Returns (meta, "ok") on success or (partial_meta, "extraction_unclear") on failure.
    Never raises.
    """
    regex_data = _regex_extract(full_text)
    last_error = ""

    for attempt in range(1, 4):  # 1 initial + 2 retries = 3 total
        try:
            meta = await _llm_extract(full_text, doc_id, attempt, last_error)
            # Backfill regex fields if LLM left them empty
            if not meta.monetary_terms:
                object.__setattr__(meta, "monetary_terms", regex_data["monetary_terms"])
            if not meta.exhibits_referenced:
                object.__setattr__(meta, "exhibits_referenced", regex_data["exhibits_referenced"])
            if not meta.signature_blocks:
                object.__setattr__(meta, "signature_blocks", regex_data["signature_blocks"])
            logger.bind(doc_id=doc_id, doc_type=meta.doc_type).info("structured_extract_done")
            return meta, "ok"
        except (ValidationError, Exception) as exc:
            last_error = str(exc)[:500]
            logger.bind(doc_id=doc_id, attempt=attempt, error=last_error).warning(
                "structured_extract_retry"
            )

    # All attempts exhausted — return partial meta from regex pass
    logger.bind(doc_id=doc_id, last_error=last_error).warning("extraction_failed")
    partial = DocumentMeta(
        doc_id=doc_id,
        doc_type="other",
        parties=[],
        effective_date=None,
        monetary_terms=list(regex_data["monetary_terms"]),  # type: ignore[arg-type]
        defined_terms=[],
        exhibits_referenced=list(regex_data["exhibits_referenced"]),  # type: ignore[arg-type]
        signature_blocks=list(regex_data["signature_blocks"]),  # type: ignore[arg-type]
        governing_law=None,
        confidence=0.0,
    )
    return partial, "extraction_unclear"
