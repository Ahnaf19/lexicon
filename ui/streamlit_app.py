"""Lexicon Streamlit UI — single entrypoint (PRD §9).

Run: uv run streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

import theme as _theme
from components import banner
from views import overview, patterns, viewer

# ─── Page config (must be first Streamlit call) ──────────────────────────────

st.set_page_config(
    page_title="Lexicon",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme management ────────────────────────────────────────────────────────

if "theme" not in st.session_state:
    st.session_state["theme"] = "parchment"

_theme.render_theme(st.session_state["theme"])

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    banner()

    st.markdown(
        '<p style="font-family:\'Inter\',sans-serif;font-size:0.7rem;'
        'font-weight:600;letter-spacing:0.08em;color:var(--text-muted);'
        'text-transform:uppercase;margin:0 0 0.4rem">Navigate</p>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["Case Overview", "Checklist Viewer", "Patterns Inspector"],
        label_visibility="collapsed",
        index=0,
    )

    st.divider()

    st.markdown(
        '<p style="font-family:\'Inter\',sans-serif;font-size:0.7rem;'
        'font-weight:600;letter-spacing:0.08em;color:var(--text-muted);'
        'text-transform:uppercase;margin:0 0 0.4rem">Theme</p>',
        unsafe_allow_html=True,
    )
    theme_choice = st.radio(
        "Theme",
        ["Parchment ☀️", "Ink 🌙", "System 🖥"],
        label_visibility="collapsed",
    )
    _theme_map = {
        "Parchment ☀️": "parchment",
        "Ink 🌙": "ink",
        "System 🖥": "system",
    }
    selected_theme = _theme_map[theme_choice]
    if selected_theme != st.session_state["theme"]:
        st.session_state["theme"] = selected_theme
        st.rerun()

# ─── Query-param routing (deep links from overview → viewer) ─────────────────

qp_page = st.query_params.get("page", "")
qp_checklist = st.query_params.get("checklist_id", "")

if qp_page == "viewer":
    active_page = "Checklist Viewer"
elif qp_page == "patterns":
    active_page = "Patterns Inspector"
elif qp_page == "overview":
    active_page = "Case Overview"
else:
    active_page = page

# ─── Page dispatch ───────────────────────────────────────────────────────────

if active_page == "Case Overview":
    st.query_params.clear()
    overview.render()

elif active_page == "Checklist Viewer":
    viewer.render(checklist_id=qp_checklist)

elif active_page == "Patterns Inspector":
    st.query_params.clear()
    patterns.render()
