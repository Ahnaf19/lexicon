"""RRF merge correctness: known rankings → expected merged order, deterministic tie-break."""

from __future__ import annotations

import uuid

import pytest

from app.retrieval.hybrid_search import rrf_merge

_A = uuid.UUID("00000000-0000-0000-0000-000000000001")
_B = uuid.UUID("00000000-0000-0000-0000-000000000002")
_C = uuid.UUID("00000000-0000-0000-0000-000000000003")
_D = uuid.UUID("00000000-0000-0000-0000-000000000004")

DOC = uuid.uuid4()


def _row(chunk_id: uuid.UUID, score: float = 0.5) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": DOC,
        "page_number": 1,
        "char_offset_start": 0,
        "char_offset_end": 100,
        "text": f"text for {chunk_id}",
        "parent_section_id": None,
        "score": score,
    }


def test_rrf_top_ranked_in_both_wins() -> None:
    dense = [_row(_A), _row(_B), _row(_C)]
    sparse = [_row(_A), _row(_C), _row(_B)]
    merged = rrf_merge(dense, sparse)
    # _A is rank 1 in both — highest RRF
    assert merged[0]["chunk_id"] == _A


def test_rrf_score_formula() -> None:
    k = 60
    dense = [_row(_A)]  # rank 1
    sparse = [_row(_A)]  # rank 1
    merged = rrf_merge(dense, sparse, k_rrf=k)
    expected = 1 / (k + 1) + 1 / (k + 1)
    assert abs(merged[0]["rrf_score"] - expected) < 1e-9


def test_rrf_only_dense_hit() -> None:
    dense = [_row(_A), _row(_B)]
    sparse = [_row(_C)]
    merged = rrf_merge(dense, sparse)
    # _C appears only in sparse; _A and _B only in dense
    chunk_ids = [r["chunk_id"] for r in merged]
    assert _A in chunk_ids
    assert _C in chunk_ids


def test_rrf_deterministic_tie_break() -> None:
    # Two chunks each appear at rank 1 in one branch only → same RRF score
    # _A < _B as strings → _A should come first
    dense = [_row(_A)]  # only _A
    sparse = [_row(_B)]  # only _B
    merged = rrf_merge(dense, sparse)
    assert merged[0]["chunk_id"] == _A
    assert merged[1]["chunk_id"] == _B


def test_rrf_dense_rank_sparse_rank_populated() -> None:
    dense = [_row(_A), _row(_B)]
    sparse = [_row(_B), _row(_A)]
    merged = rrf_merge(dense, sparse)
    by_id = {r["chunk_id"]: r for r in merged}
    assert by_id[_A]["dense_rank"] == 1
    assert by_id[_A]["sparse_rank"] == 2
    assert by_id[_B]["dense_rank"] == 2
    assert by_id[_B]["sparse_rank"] == 1


def test_rrf_absent_chunk_has_none_rank() -> None:
    dense = [_row(_A)]
    sparse = [_row(_B)]
    merged = rrf_merge(dense, sparse)
    by_id = {r["chunk_id"]: r for r in merged}
    assert by_id[_A]["sparse_rank"] is None
    assert by_id[_B]["dense_rank"] is None


def test_rrf_top_k_capped() -> None:
    dense = [_row(uuid.uuid4()) for _ in range(20)]
    sparse = [_row(uuid.uuid4()) for _ in range(20)]
    merged = rrf_merge(dense, sparse)
    from app.retrieval.hybrid_search import TOP_K
    assert len(merged) <= TOP_K
