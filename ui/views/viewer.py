"""Checklist Viewer page — the money page: grouped items, edits, finalize."""
from __future__ import annotations

import uuid as _uuid

import streamlit as st

import api_client
from components import (
    confidence_bar,
    eval_metrics_chips,
    evidence_chip,
    pattern_lineage_badges,
    stats_strip,
    status_badge,
)
from theme import CATEGORY_ORDER
from utils import humanize_relative_time, parse_iso, truncate_uuid


# ─── Add-item dialog (must be module-level for @st.dialog) ───────────────────


@st.dialog("Add Item")
def _add_item_dialog() -> None:
    """Modal form for adding a new item to a checklist category."""
    checklist_id = st.session_state.get("_add_checklist_id", "")
    category = st.session_state.get("_add_category", "Other")

    st.markdown(
        f'<span class="lex-badge lex-badge-neutral">{category}</span>',
        unsafe_allow_html=True,
    )

    with st.form("add_item_form"):
        title = st.text_input("Title *")
        description = st.text_area("Description", height=72)
        status = st.selectbox(
            "Status",
            ["missing", "unclear"],
            help="Cannot set 'present' on a new item — add evidence after creation.",
        )
        required = st.checkbox("Required", value=True)
        rationale = st.text_area("Rationale", height=72)

        if st.form_submit_button("Add Item", type="primary"):
            if not title.strip():
                st.error("Title is required.")
                return
            item = {
                "id": str(_uuid.uuid4()),
                "category": category,
                "title": title.strip(),
                "description": description.strip(),
                "status": status,
                "required": required,
                "confidence": 0.5,
                "rationale": rationale.strip(),
                "evidence": [],
                "learned_from_pattern_ids": [],
            }
            try:
                api_client.add_item(checklist_id, item)
                api_client.get_checklist.clear()
                api_client.get_case_checklists.clear()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


# ─── Item card ────────────────────────────────────────────────────────────────


def _edit_form(item: dict, checklist_id: str, item_id: str) -> None:
    """Inline edit form inside a st.popover."""
    with st.form(key=f"edit_{item_id}"):
        new_title = st.text_input("Title", value=item.get("title", ""))
        new_desc = st.text_area("Description", value=item.get("description", ""), height=72)

        has_evidence = bool(item.get("evidence", []))
        status_opts = ["missing", "unclear"] + (["present"] if has_evidence else [])
        cur_status = item.get("status", "missing")
        if cur_status not in status_opts:
            cur_status = status_opts[0]

        new_status = st.selectbox("Status", status_opts, index=status_opts.index(cur_status))
        if not has_evidence:
            st.caption("Add evidence first to mark as 'present'.")

        new_rationale = st.text_area("Rationale", value=item.get("rationale", ""), height=72)

        if st.form_submit_button("Save changes", type="primary"):
            fields: dict[str, object] = {}
            if new_title.strip() != item.get("title", ""):
                fields["title"] = new_title.strip()
            if new_desc.strip() != item.get("description", ""):
                fields["description"] = new_desc.strip()
            if new_status != item.get("status"):
                fields["status"] = new_status
            if new_rationale.strip() != item.get("rationale", ""):
                fields["rationale"] = new_rationale.strip()
            try:
                if fields:
                    api_client.patch_item(checklist_id, item_id, fields)
                api_client.get_checklist.clear()
                api_client.get_case_checklists.clear()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def _item_card(item: dict, checklist_id: str, all_patterns: list[dict]) -> None:
    """Render one checklist item as a bordered card."""
    item_id = str(item.get("id", ""))
    status = item.get("status", "missing")

    with st.container(border=True):
        # ── Header row: title + badge + confidence ──
        col_title, col_conf = st.columns([5, 1])
        with col_title:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap">'
                f'<strong style="font-family:\'Fraunces\',Georgia,serif;'
                f'font-size:1rem;color:var(--text)">'
                f"{item.get('title', 'Untitled')}"
                f"</strong>"
                f" {status_badge(status)}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_conf:
            conf = item.get("confidence") or 0.0
            st.markdown(confidence_bar(conf), unsafe_allow_html=True)

        # ── Rationale ──
        rationale = item.get("rationale", "")
        if rationale:
            st.markdown(
                f'<p style="color:var(--text-muted);font-size:0.855rem;'
                f'font-style:italic;margin:0.35rem 0 0.5rem">{rationale}</p>',
                unsafe_allow_html=True,
            )

        # ── Evidence chips ──
        for cit in item.get("evidence", []):
            evidence_chip(cit)

        # ── Pattern lineage badges ──
        pattern_ids = item.get("learned_from_pattern_ids", [])
        if pattern_ids:
            st.markdown(
                f'<span class="lex-badge lex-badge-learning" '
                f'style="margin-bottom:0.4rem;display:inline-block">'
                f"Auto-applied · {len(pattern_ids)} pattern"
                f'{"s" if len(pattern_ids) != 1 else ""}'
                f"</span>",
                unsafe_allow_html=True,
            )
            pattern_lineage_badges(pattern_ids, all_patterns)

        # ── Action row ──
        col_edit, col_del, _ = st.columns([1, 1, 6])
        with col_edit:
            with st.popover("Edit"):
                _edit_form(item, checklist_id, item_id)
        with col_del:
            with st.popover("Delete"):
                st.warning("Permanently delete this item?")
                if st.button("Confirm", key=f"del_confirm_{item_id}", type="primary"):
                    try:
                        api_client.delete_item(checklist_id, item_id)
                        api_client.get_checklist.clear()
                        api_client.get_case_checklists.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))


# ─── Page renderer ────────────────────────────────────────────────────────────


def render(checklist_id: str) -> None:
    """Render the Checklist Viewer page."""
    # ── Checklist ID input (if not provided via query param) ──
    if not checklist_id:
        st.markdown("### Open a Checklist")
        col_in, col_btn = st.columns([4, 1])
        with col_in:
            cid_input = st.text_input(
                "Checklist ID",
                placeholder="Enter a checklist UUID…",
                label_visibility="collapsed",
            )
        with col_btn:
            if st.button("Load", type="primary", use_container_width=True) and cid_input:
                st.query_params.update(page="viewer", checklist_id=cid_input.strip())
                st.rerun()
        return

    # ── Load checklist ──
    try:
        checklist = api_client.get_checklist(checklist_id)
    except Exception as exc:
        st.error(f"Could not load checklist: {exc}")
        return

    items: list[dict] = checklist.get("items", [])

    # ── Header ──
    col_meta, col_btn = st.columns([5, 1])
    with col_meta:
        dt = parse_iso(checklist.get("generated_at"))
        when = humanize_relative_time(dt) if dt else "unknown"
        model_v = checklist.get("model_version") or "?"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;'
            f'margin-bottom:0.4rem">'
            f'<code style="font-size:0.85rem;color:var(--text)">'
            f'<abbr title="{checklist_id}" style="text-decoration:none">'
            f"{truncate_uuid(checklist_id)}</abbr></code>"
            f'<span style="color:var(--text-muted);font-size:0.8rem">'
            f"Generated {when}</span>"
            f'<span class="lex-badge lex-badge-neutral">{model_v}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        eval_m = checklist.get("eval_metrics")
        if eval_m:
            eval_metrics_chips(eval_m)

    with col_btn:
        if st.button("Finalize Checklist", type="primary", use_container_width=True):
            try:
                api_client.finalize_checklist(checklist_id)
                api_client.get_checklist.clear()
                api_client.get_case_checklists.clear()
                st.toast("Pattern extraction triggered")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    # ── Stats strip ──
    stats_strip(items)

    # ── Search filter ──
    search = st.text_input(
        "Filter items",
        placeholder="Search by title…",
        label_visibility="collapsed",
    )

    # ── Fetch pattern data for lineage badges (best-effort) ──
    all_patterns: list[dict] = []
    try:
        all_patterns = api_client.get_patterns()
    except Exception:
        pass

    # ── Group by canonical category order ──
    categorized: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_ORDER}
    for item in items:
        cat = item.get("category", "Other")
        if cat not in categorized:
            cat = "Other"
        if search and search.lower() not in item.get("title", "").lower():
            continue
        categorized[cat].append(item)

    for cat in CATEGORY_ORDER:
        cat_items = categorized[cat]
        if not cat_items:
            continue

        with st.expander(f"**{cat}** ({len(cat_items)})", expanded=True):
            for item in cat_items:
                _item_card(item, checklist_id, all_patterns)

            # Add Item button
            if st.button(
                f"+ Add item to {cat}",
                key=f"add_{cat}_{checklist_id}",
                use_container_width=True,
            ):
                st.session_state["_add_checklist_id"] = checklist_id
                st.session_state["_add_category"] = cat
                _add_item_dialog()
