"""Hybrid pgvector + tsvector search with RRF merge and parent expansion (PRD §5d).

Pipeline:
  dense()             — cosine similarity via pgvector HNSW
  sparse()            — ts_rank_cd over tsvector GIN index
  rrf_merge()         — Reciprocal Rank Fusion (k=60), top-8 after fusion
  expand_with_parents() — fetch parent section texts, pack to 3500-token budget
  search()            — public entry point returning list[EvidenceCitation]
"""

from __future__ import annotations

import uuid
from typing import Any

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel

from app.models.pydantic_models import EvidenceCitation
from app.retrieval.chunking import _ENCODER
from app.retrieval.embedding import embed_query

# ---------------------------------------------------------------------------
# SearchHit — EvidenceCitation + ranking metadata for CLI / generation use
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    """EvidenceCitation fields plus RRF rank provenance."""

    citation_id: uuid.UUID
    chunk_id: uuid.UUID
    doc_id: uuid.UUID
    page_number: int
    char_offset_start: int
    char_offset_end: int
    snippet: str
    retrieval_score: float
    rerank_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None

    def to_evidence_citation(self) -> EvidenceCitation:
        return EvidenceCitation(
            citation_id=self.citation_id,
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            page_number=self.page_number,
            char_offset_start=self.char_offset_start,
            char_offset_end=self.char_offset_end,
            snippet=self.snippet,
            retrieval_score=self.retrieval_score,
            rerank_score=self.rerank_score,
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_PER_BRANCH: int = 20
K_RRF: int = 60
TOP_K: int = 8
PARENT_BUDGET_TOKENS: int = 3500


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------


async def dense(
    query: str,
    case_id: uuid.UUID,
    session: AsyncSession,
    k: int = K_PER_BRANCH,
) -> list[dict[str, Any]]:
    """Return top-k window chunks ranked by cosine similarity."""
    qvec = await embed_query(query)
    # pgvector operator <=> = cosine distance; score = 1 - distance
    sql = text(
        """
        SELECT
            c.id            AS chunk_id,
            c.doc_id        AS doc_id,
            c.page_number   AS page_number,
            c.char_offset_start,
            c.char_offset_end,
            c.text          AS text,
            c.parent_section_id,
            1.0 - (c.embedding <=> CAST(:qvec AS vector)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        WHERE
            c.meta->>'kind' = 'window'
            AND c.embedding IS NOT NULL
            AND d.case_id   = :case_id
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT :k
        """
    )
    result = await session.execute(
        sql,
        {"qvec": str(qvec), "case_id": str(case_id), "k": k},
    )
    return [dict(r._mapping) for r in result]


# ---------------------------------------------------------------------------
# Sparse retrieval
# ---------------------------------------------------------------------------


async def sparse(
    query: str,
    case_id: uuid.UUID,
    session: AsyncSession,
    k: int = K_PER_BRANCH,
) -> list[dict[str, Any]]:
    """Return top-k window chunks ranked by ts_rank_cd."""
    sql = text(
        """
        SELECT
            c.id            AS chunk_id,
            c.doc_id        AS doc_id,
            c.page_number   AS page_number,
            c.char_offset_start,
            c.char_offset_end,
            c.text          AS text,
            c.parent_section_id,
            ts_rank_cd(c.tsv, plainto_tsquery('english', :q)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        WHERE
            c.meta->>'kind' = 'window'
            AND c.embedding IS NOT NULL
            AND d.case_id   = :case_id
            AND c.tsv @@ plainto_tsquery('english', :q)
        ORDER BY score DESC
        LIMIT :k
        """
    )
    result = await session.execute(
        sql,
        {"q": query, "case_id": str(case_id), "k": k},
    )
    return [dict(r._mapping) for r in result]


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------


def rrf_merge(
    dense_rows: list[dict[str, Any]],
    sparse_rows: list[dict[str, Any]],
    k_rrf: int = K_RRF,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over dense + sparse ranked lists.

    Returns rows sorted by descending RRF score, top-TOP_K.
    Tie-break is ascending str(chunk_id) for determinism.
    Each returned dict has the original row fields plus:
      rrf_score, dense_rank (1-based, None if absent), sparse_rank (1-based, None if absent)
    """
    dense_rank: dict[uuid.UUID, int] = {
        r["chunk_id"]: i + 1 for i, r in enumerate(dense_rows)
    }
    sparse_rank: dict[uuid.UUID, int] = {
        r["chunk_id"]: i + 1 for i, r in enumerate(sparse_rows)
    }

    # Union of all unique chunk ids
    all_ids: set[uuid.UUID] = set(dense_rank) | set(sparse_rank)

    # Build a lookup for row data (prefer dense for text/meta)
    row_by_id: dict[uuid.UUID, dict[str, Any]] = {}
    for r in dense_rows:
        row_by_id[r["chunk_id"]] = r
    for r in sparse_rows:
        if r["chunk_id"] not in row_by_id:
            row_by_id[r["chunk_id"]] = r

    scored: list[dict[str, Any]] = []
    for cid in all_ids:
        dr = dense_rank.get(cid)
        sr = sparse_rank.get(cid)
        rrf = 0.0
        if dr is not None:
            rrf += 1.0 / (k_rrf + dr)
        if sr is not None:
            rrf += 1.0 / (k_rrf + sr)
        row = dict(row_by_id[cid])
        row["rrf_score"] = rrf
        row["dense_rank"] = dr
        row["sparse_rank"] = sr
        scored.append(row)

    scored.sort(key=lambda r: (-r["rrf_score"], str(r["chunk_id"])))
    return scored[:TOP_K]


# ---------------------------------------------------------------------------
# Parent expansion
# ---------------------------------------------------------------------------


async def expand_with_parents(
    matches: list[dict[str, Any]],
    session: AsyncSession,
    budget_tokens: int = PARENT_BUDGET_TOKENS,
) -> list[SearchHit]:
    """Fetch parent section texts for matched windows; pack greedily to budget.

    Returns list[SearchHit] with full rank provenance.
    """
    # Collect unique parent_section_ids
    parent_ids = list(
        {r["parent_section_id"] for r in matches if r.get("parent_section_id")}
    )

    parent_text: dict[uuid.UUID, str] = {}
    if parent_ids:
        result = await session.execute(
            text("SELECT id, text FROM chunks WHERE id = ANY(:ids)"),
            {"ids": [str(p) for p in parent_ids]},
        )
        for pid_str, txt in result:
            parent_text[uuid.UUID(str(pid_str))] = txt

    hits: list[SearchHit] = []
    tokens_used = 0

    for rank_idx, row in enumerate(matches):
        pid = row.get("parent_section_id")
        context = parent_text.get(pid, row["text"]) if pid else row["text"]
        token_cost = len(_ENCODER.encode(context))

        if tokens_used + token_cost > budget_tokens and hits:
            logger.bind(rank=rank_idx, tokens_used=tokens_used).debug(
                "parent_expand_budget_reached"
            )
            break

        hits.append(
            SearchHit(
                citation_id=uuid.uuid4(),
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                page_number=row["page_number"],
                char_offset_start=row["char_offset_start"],
                char_offset_end=row["char_offset_end"],
                snippet=row["text"][:300],
                retrieval_score=row["rrf_score"],
                rerank_score=None,
                dense_rank=row.get("dense_rank"),
                sparse_rank=row.get("sparse_rank"),
            )
        )
        tokens_used += token_cost

    return hits


# ---------------------------------------------------------------------------
# Public search entry point
# ---------------------------------------------------------------------------


async def search(
    query: str,
    case_id: uuid.UUID,
    session: AsyncSession,
    k: int = TOP_K,
) -> list[SearchHit]:
    """Hybrid dense + sparse search with RRF merge and parent-section expansion.

    Returns list[SearchHit]; call .to_evidence_citation() for each hit as needed
    by the generation layer (PRD §5d).
    """
    import asyncio

    dense_rows, sparse_rows = await asyncio.gather(
        dense(query, case_id, session, k=K_PER_BRANCH),
        sparse(query, case_id, session, k=K_PER_BRANCH),
    )
    logger.bind(
        query=query[:80],
        case_id=str(case_id),
        dense_hits=len(dense_rows),
        sparse_hits=len(sparse_rows),
    ).debug("hybrid_search_branches")

    merged = rrf_merge(dense_rows, sparse_rows)
    return await expand_with_parents(merged, session)
