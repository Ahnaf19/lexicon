"""Patterns Inspector page — browse, filter, sort, and dismiss learned patterns."""
from __future__ import annotations

import streamlit as st

import api_client
from components import confidence_bar
from theme import PATTERN_TYPES
from utils import humanize_relative_time, parse_iso, truncate_uuid


def render() -> None:
    """Render the Patterns Inspector page."""
    st.markdown("### Learned Patterns")

    # ── Filter / sort controls ──
    col_dt, col_pt, col_prom, col_sort = st.columns([2, 2, 1, 2])
    with col_dt:
        doc_type_filter = st.text_input(
            "Doc type",
            placeholder="e.g. nda",
            label_visibility="visible",
        )
    with col_pt:
        pt_options = ["(all)"] + list(PATTERN_TYPES)
        pt_filter = st.selectbox("Pattern type", pt_options, label_visibility="visible")
    with col_prom:
        promoted_only = st.toggle("Promoted only", value=False)
    with col_sort:
        sort_by = st.selectbox(
            "Sort",
            ["confidence ↓", "corroboration ↓", "created ↓"],
            label_visibility="visible",
        )

    # ── Fetch ──
    try:
        patterns = api_client.get_patterns(
            doc_type=doc_type_filter.strip() or None,
            promoted=True if promoted_only else None,
            pattern_type=None if pt_filter == "(all)" else pt_filter,
        )
    except Exception as exc:
        st.error(f"Could not load patterns: {exc}")
        return

    # ── Sort client-side ──
    sort_key_map = {
        "confidence ↓": lambda p: -(p.get("confidence") or 0.0),
        "corroboration ↓": lambda p: -(p.get("corroborating_edit_count") or 0),
        "created ↓": lambda p: p.get("created_at", ""),
    }
    patterns = sorted(patterns, key=sort_key_map[sort_by])

    # ── Empty state ──
    if not patterns:
        st.markdown(
            '<div class="lex-card" style="text-align:center;padding:2.5rem 1rem">'
            '<p style="color:var(--text-muted);font-family:\'Inter\',sans-serif;margin:0">'
            "No patterns learned yet — finalize a checklist to start teaching Lexicon."
            "<br><span style='color:var(--accent)'>↑ Go to Case Overview to finalize.</span>"
            "</p></div>",
            unsafe_allow_html=True,
        )
        return

    # ── Table header ──
    st.markdown(
        f'<p style="color:var(--text-muted);font-size:0.8rem;margin:0 0 0.5rem">'
        f"{len(patterns)} pattern{'s' if len(patterns) != 1 else ''}</p>",
        unsafe_allow_html=True,
    )

    # ── Rows ──
    header = st.columns([2, 1.5, 1, 1, 0.8, 1.5, 0.9])
    for h, label in zip(
        header,
        ["Pattern type", "Doc scope", "Confidence", "Corroboration", "Promoted", "Created", ""],
    ):
        h.markdown(
            f'<span style="font-size:0.75rem;font-weight:600;color:var(--text-muted);'
            f'font-family:\'Inter\',sans-serif">{label}</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<hr style="margin:0.3rem 0 0.4rem;border-color:var(--border)">',
        unsafe_allow_html=True,
    )

    for p in patterns:
        pid = str(p.get("id", ""))
        pt = p.get("pattern_type", "unknown")
        scope = p.get("doc_type_scope", "any")
        conf = p.get("confidence") or 0.0
        corr = p.get("corroborating_edit_count") or 0
        promoted = p.get("promoted", False)
        dt = parse_iso(p.get("created_at"))
        when = humanize_relative_time(dt) if dt else "?"

        cols = st.columns([2, 1.5, 1, 1, 0.8, 1.5, 0.9])

        # Pattern type chip
        cols[0].markdown(
            f'<span class="lex-badge lex-badge-learning">{pt}</span>',
            unsafe_allow_html=True,
        )

        # Doc type scope
        cols[1].markdown(
            f'<code style="font-size:0.78rem;color:var(--text)">{scope}</code>',
            unsafe_allow_html=True,
        )

        # Confidence bar
        cols[2].markdown(confidence_bar(conf), unsafe_allow_html=True)

        # Corroboration mini-bar
        max_corr = max((p2.get("corroborating_edit_count") or 0 for p2 in patterns), default=1) or 1
        corr_pct = int((corr / max_corr) * 100)
        cols[3].markdown(
            f'<span style="display:inline-flex;align-items:center;gap:0.35rem">'
            f'<span class="lex-conf-track">'
            f'<span class="lex-conf-fill" style="width:{corr_pct}%;'
            f'background:var(--learning)"></span></span>'
            f'<span class="lex-conf-label">{corr}</span>'
            f"</span>",
            unsafe_allow_html=True,
        )

        # Promoted
        cols[4].markdown(
            f'<span style="color:{"var(--accent)" if promoted else "var(--text-muted)"};'
            f'font-size:1rem">{"✓" if promoted else "·"}</span>',
            unsafe_allow_html=True,
        )

        # Created
        cols[5].markdown(
            f'<span style="font-size:0.78rem;color:var(--text-muted)">{when}</span>',
            unsafe_allow_html=True,
        )

        # Dismiss
        if cols[6].button("Dismiss", key=f"dismiss_{pid}", use_container_width=True):
            try:
                api_client.dismiss_pattern(pid)
                api_client.get_patterns.clear()
                st.toast("Pattern dismissed")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        # Rule detail expander
        rule_json = p.get("rule_json", {})
        if rule_json:
            with st.expander(f"rule_json · {truncate_uuid(pid)}", expanded=False):
                st.json(rule_json)

        st.markdown(
            '<hr style="margin:0.2rem 0;border-color:var(--border);opacity:0.5">',
            unsafe_allow_html=True,
        )
