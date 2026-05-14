"""Verify bisect-based offset→source-block lookup matches brute-force linear scan (P5)."""

from __future__ import annotations

import bisect
import random

from app.retrieval.chunking import _build_cumulative_offsets


def _linear_scan(offsets: list[int], pos: int) -> int:
    """Brute-force: find first block whose cumulative end >= pos."""
    for i, end in enumerate(offsets):
        if end >= pos:
            return i
    return len(offsets) - 1


def test_bisect_matches_linear_scan_on_fixture() -> None:
    random.seed(42)
    block_texts = [
        f"Block {i}: " + "x" * random.randint(10, 200)
        for i in range(50)
    ]
    offsets = _build_cumulative_offsets(block_texts)

    doc_len = offsets[-1]
    test_positions = [random.randint(0, doc_len - 1) for _ in range(20)]

    for pos in test_positions:
        bisect_result = bisect.bisect_left(offsets, pos)
        bisect_result = min(bisect_result, len(offsets) - 1)
        linear_result = _linear_scan(offsets, pos)
        assert bisect_result == linear_result, (
            f"Mismatch at pos={pos}: bisect={bisect_result}, linear={linear_result}"
        )


def test_offsets_monotonically_increasing() -> None:
    texts = ["hello", "world", "foo bar", "baz"]
    offsets = _build_cumulative_offsets(texts)
    for i in range(len(offsets) - 1):
        assert offsets[i] < offsets[i + 1]


def test_last_offset_equals_total_length_with_separators() -> None:
    texts = ["abc", "de", "fghi"]
    offsets = _build_cumulative_offsets(texts)
    # Each block's len + 1 for "\n" separator
    expected = sum(len(t) + 1 for t in texts)
    assert offsets[-1] == expected
