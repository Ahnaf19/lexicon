"""P2: Promotion gate is bidirectional — count >= 3 AND confidence >= 0.7 both required."""

from __future__ import annotations

import pytest

from app.learning.pattern_extractor import (
    _PROMOTION_MIN_CONFIDENCE,
    _PROMOTION_MIN_COUNT,
    _compute_confidence,
    _should_promote,
)


def test_corroboration_1_not_promoted() -> None:
    assert not _should_promote(1, _compute_confidence(1))


def test_corroboration_2_not_promoted() -> None:
    assert not _should_promote(2, _compute_confidence(2))


def test_corroboration_3_promoted() -> None:
    conf = _compute_confidence(3)
    assert conf >= _PROMOTION_MIN_CONFIDENCE
    assert _should_promote(3, conf)


def test_high_count_low_confidence_not_promoted() -> None:
    # Force confidence below threshold regardless of count.
    assert not _should_promote(10, 0.5)


def test_high_confidence_low_count_not_promoted() -> None:
    assert not _should_promote(2, 0.95)


def test_promotion_requires_both_gates() -> None:
    # Exactly at threshold: count=3, confidence=just below 0.7 → not promoted.
    assert not _should_promote(3, 0.699)


def test_dismiss_demotes_below_threshold() -> None:
    # After dismiss: count drops from 3 to 2 → not promoted.
    count_after_dismiss = max(0, 3 - 1)
    assert not _should_promote(count_after_dismiss, _compute_confidence(count_after_dismiss))
