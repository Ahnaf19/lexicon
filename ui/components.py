"""Reusable Streamlit render helpers — pure presentation, no HTTP calls.

All functions write directly to Streamlit's current container. HTML-returning
helpers (status_badge, confidence_bar) return a string for inline embedding in
larger HTML blocks.
"""
from __future__ import annotations

import streamlit as st

from theme import STATUS_STYLES


def banner() -> None:
    """Render the gradient Lexicon wordmark banner."""
    st.markdown(
        """
        <div class="lexicon-banner">
          <p class="lex-wordmark">Lex<span class="lex-accent">icon</span></p>
          <p class="lex-tagline">Grounded checklists for messy documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    """Return an inline HTML badge pill for the given item status."""
    s = STATUS_STYLES.get(status, STATUS_STYLES["missing"])
    return (
        f'<span class="lex-badge {s["cls"]}">'
        f'{s["glyph"]}&nbsp;{s["label"]}</span>'
    )


def confidence_bar(value: float) -> str:
    """Return HTML for a labelled confidence progress bar."""
    pct = max(0, min(100, int(value * 100)))
    return (
        f'<span class="lex-conf-wrap">'
        f'<span class="lex-conf-track">'
        f'<span class="lex-conf-fill" style="width:{pct}%"></span>'
        f'</span>'
        f'<span class="lex-conf-label">{pct}%</span>'
        f'</span>'
    )


def stats_strip(items: list[dict]) -> None:  # type: ignore[type-arg]
    """Render a one-line summary strip with item counts, avg confidence, and a histogram."""
    total = len(items)
    if not total:
        return
    present = sum(1 for i in items if i.get("status") == "present")
    unclear = sum(1 for i in items if i.get("status") == "unclear")
    missing = sum(1 for i in items if i.get("status") == "missing")
    avg_conf = sum(i.get("confidence", 0.0) for i in items) / total

    buckets = [0] * 10
    for item in items:
        b = min(9, int((item.get("confidence") or 0.0) * 10))
        buckets[b] += 1
    max_b = max(buckets) or 1
    hist_html = "".join(
        f'<span class="lex-hist-bar" style="height:{max(4, int(b / max_b * 24))}px" '
        f'title="{i * 10}–{(i + 1) * 10}%"></span>'
        for i, b in enumerate(buckets)
    )

    st.markdown(
        f"""
        <div class="lex-stats">
          <span>
            <span class="stat-val" style="color:var(--status-present)">{present}</span>
            &nbsp;present
          </span>
          <span>
            <span class="stat-val" style="color:var(--status-unclear)">{unclear}</span>
            &nbsp;unclear
          </span>
          <span>
            <span class="stat-val" style="color:var(--status-missing)">{missing}</span>
            &nbsp;missing
          </span>
          <span>
            avg confidence&nbsp;
            <span class="stat-val">{avg_conf:.0%}</span>
          </span>
          <span style="display:flex;align-items:flex-end;height:28px">{hist_html}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def eval_metrics_chips(metrics: dict[str, float]) -> None:
    """Render eval metric pills (iris-tinted) when Checklist.eval_metrics is present."""
    if not metrics:
        return
    chips = "".join(
        f'<span class="lex-badge lex-badge-learning" style="margin-right:0.35rem">'
        f'{k}: <strong>{v:.3f}</strong></span>'
        for k, v in metrics.items()
    )
    st.markdown(f'<div style="margin:0.4rem 0">{chips}</div>', unsafe_allow_html=True)


def skeleton(height: int = 56) -> None:
    """Render a shimmer skeleton placeholder for cache-cold loads."""
    st.markdown(
        f'<div class="lex-skeleton" style="height:{height}px;margin-bottom:0.75rem"></div>',
        unsafe_allow_html=True,
    )


def evidence_chip(citation: dict) -> None:  # type: ignore[type-arg]
    """Render an expandable evidence citation chip with snippet + provenance."""
    page = citation.get("page_number", "?")
    doc_id_short = str(citation.get("doc_id", ""))[:8]
    snippet = citation.get("snippet", "")
    score = citation.get("retrieval_score") or 0.0
    label = f"[doc={doc_id_short} p.{page}]  score {score:.2f}"
    with st.expander(label):
        st.markdown(f"**Page {page}** · retrieval score `{score:.3f}`")
        if citation.get("rerank_score") is not None:
            st.markdown(f"rerank score `{citation['rerank_score']:.3f}`")
        st.markdown(f"> {snippet}")


def pattern_lineage_badges(pattern_ids: list, all_patterns: list[dict]) -> None:  # type: ignore[type-arg]
    """Render iris-tinted Auto-applied chips, each with a popover showing rule details."""
    if not pattern_ids:
        return
    pattern_map = {str(p["id"]): p for p in all_patterns}
    for pid in pattern_ids:
        pid_str = str(pid)
        with st.popover(
            f"Auto-applied · {pid_str[:8]}",
        ):
            p = pattern_map.get(pid_str)
            if p:
                st.markdown(
                    f'<span class="lex-badge lex-badge-learning">'
                    f'{p.get("pattern_type", "unknown")}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Doc scope:** `{p.get('doc_type_scope', 'any')}`")
                st.markdown(
                    f"Confidence `{p.get('confidence', 0):.2f}` · "
                    f"corroboration `{p.get('corroborating_edit_count', 0)}`"
                )
                st.json(p.get("rule_json", {}))
            else:
                st.caption("Pattern details not available.")
