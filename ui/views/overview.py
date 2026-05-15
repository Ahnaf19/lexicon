"""Case Overview page — upload docs, generate checklists, browse cases."""
from __future__ import annotations

import time
import uuid as _uuid

import streamlit as st

import api_client
from utils import humanize_relative_time, parse_iso, truncate_uuid


def _generate_form() -> None:
    """Generate Checklist card with SSE progress stream."""
    st.markdown("### Generate Checklist")
    col1, col2 = st.columns([4, 1])
    with col1:
        case_id = st.text_input(
            "Case ID",
            value=st.session_state.get("active_case_id", ""),
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            help="A UUID grouping documents from the same matter.",
            key="gen_case_id",
        )
    with col2:
        st.write("")
        if st.button("New UUID", key="new_uuid_gen", use_container_width=True):
            st.session_state["active_case_id"] = str(_uuid.uuid4())
            st.rerun()

    template = st.selectbox(
        "Template",
        ["(auto-detect)", "commercial_contract", "nda"],
        help="Leave blank to let Lexicon classify the document set automatically.",
    )

    if st.button("Generate", type="primary", use_container_width=False):
        cid = st.session_state.get("gen_case_id", case_id).strip()
        if not cid:
            st.error("Case ID is required.")
            return
        slug = None if template == "(auto-detect)" else template
        st.session_state["active_case_id"] = cid

        checklist_id = None
        error_detail = None
        events: dict[str, str] = {}

        with st.status("Generating checklist…", expanded=True) as gen_status:
            log_ph = st.empty()
            try:
                for event_name, data in api_client.stream_generate(cid, slug):
                    node = data.get("node", "")
                    if event_name == "node_start" and node:
                        events[node] = "running"
                    elif event_name == "node_end" and node:
                        events[node] = "done"
                    elif event_name == "done":
                        checklist_id = data.get("checklist_id")
                        gen_status.update(label="Checklist generated!", state="complete")
                        break
                    elif event_name == "error":
                        error_detail = data.get("detail", "Generation failed")
                        gen_status.update(label="Generation failed", state="error")
                        break

                    lines = [
                        f"{'✓' if s == 'done' else '▸'} `{n}`"
                        for n, s in events.items()
                    ]
                    log_ph.markdown("\n\n".join(lines) if lines else "Starting…")

            except Exception as exc:
                error_detail = str(exc)
                gen_status.update(label="Connection error", state="error")

        if error_detail:
            st.error(error_detail)
        if checklist_id:
            api_client.get_checklist.clear()
            api_client.get_case_checklists.clear()
            st.query_params.update(page="viewer", checklist_id=checklist_id)
            st.rerun()


def _upload_card() -> None:
    """Document upload card with async status polling."""
    st.markdown("### Upload Document")
    col1, col2 = st.columns([4, 1])
    with col1:
        upload_case = st.text_input(
            "Case ID for upload",
            value=st.session_state.get("active_case_id", ""),
            key="upload_case_id",
            help="Documents will be grouped under this case UUID.",
        )
    with col2:
        st.write("")
        if st.button("New UUID", key="new_uuid_upload", use_container_width=True):
            st.session_state["active_case_id"] = str(_uuid.uuid4())
            st.rerun()

    uploaded = st.file_uploader(
        "PDF or image",
        type=["pdf", "jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded and st.button("Upload", type="primary"):
        cid = st.session_state.get("upload_case_id", upload_case).strip()
        if not cid:
            cid = str(_uuid.uuid4())
            st.session_state["active_case_id"] = cid

        try:
            resp = api_client.upload_document(
                file_bytes=uploaded.read(),
                filename=uploaded.name,
                mime=uploaded.type or "application/pdf",
                case_id=cid,
            )
            doc_id = resp["document_id"]
            st.session_state["active_case_id"] = cid

            with st.status(f"Processing {uploaded.name}…", expanded=True) as up_status:
                terminal = False
                for _ in range(15):  # max 30s
                    time.sleep(2)
                    api_client.get_document_status.clear()
                    try:
                        s = api_client.get_document_status(doc_id).get("status", "pending")
                    except Exception:
                        s = "pending"
                    up_status.update(label=f"Status: {s}")
                    if s in ("indexed", "error", "failed"):
                        state = "complete" if s == "indexed" else "error"
                        label = f"{'✓' if s == 'indexed' else '✗'} {uploaded.name} — {s}"
                        up_status.update(label=label, state=state)
                        terminal = True
                        break
                if not terminal:
                    up_status.update(
                        label="Still processing — check status manually",
                        state="running",
                    )
                    st.warning(
                        f"Ingestion is taking longer than expected. "
                        f"Check `/documents/{doc_id}/status` manually."
                    )

            api_client.get_cases.clear()
            api_client.get_case_documents.clear()
            st.toast(f"Uploaded {uploaded.name}")
            st.rerun()

        except Exception as exc:
            st.error(f"Upload failed: {exc}")


def _cases_grid() -> None:
    """Render the 2-column cases grid."""
    st.markdown("### Cases")
    try:
        cases = api_client.get_cases()
    except Exception as exc:
        st.warning(f"Could not load cases: {exc}")
        return

    if not cases:
        st.markdown(
            '<div class="lex-card" style="text-align:center;padding:2.5rem 1rem">'
            '<p style="color:var(--text-muted);font-family:\'Inter\',sans-serif;margin:0">'
            "No cases yet — upload a document above to get started."
            "</p></div>",
            unsafe_allow_html=True,
        )
        return

    cols = st.columns(2)
    for i, case in enumerate(cases):
        with cols[i % 2]:
            cid = str(case.get("case_id", ""))
            doc_count = case.get("doc_count", 0)

            st.markdown(
                f'<div class="lex-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'margin-bottom:0.6rem">'
                f'<code style="font-size:0.85rem;color:var(--text)">'
                f'<abbr title="{cid}" style="text-decoration:none">{truncate_uuid(cid)}</abbr>'
                f"</code>"
                f'<span class="lex-badge lex-badge-action">'
                f'{doc_count} doc{"s" if doc_count != 1 else ""}'
                f"</span></div>",
                unsafe_allow_html=True,
            )

            # Documents list
            try:
                docs = api_client.get_case_documents(cid)
                if docs:
                    st.caption("Documents")
                    for doc in docs[:4]:
                        status = doc.get("status", "pending")
                        color = {
                            "indexed": "var(--status-present)",
                            "processing": "var(--status-unclear)",
                            "pending": "var(--text-muted)",
                            "queued": "var(--text-muted)",
                        }.get(status, "var(--text-muted)")
                        dc1, dc2 = st.columns([3, 1])
                        dc1.caption(doc.get("original_filename", "?"))
                        dc2.markdown(
                            f'<span style="font-size:0.72rem;color:{color}">{status}</span>',
                            unsafe_allow_html=True,
                        )
                    if len(docs) > 4:
                        st.caption(f"+ {len(docs) - 4} more")
            except Exception:
                st.caption("Documents unavailable.")

            # Checklists list
            try:
                clists = api_client.get_case_checklists(cid)
                if clists:
                    st.caption("Checklists")
                    for cl in clists[:3]:
                        cl_id = str(cl.get("checklist_id", ""))
                        dt = parse_iso(cl.get("generated_at"))
                        when = humanize_relative_time(dt) if dt else "?"
                        item_count = cl.get("item_count", 0)
                        cc1, cc2 = st.columns([3, 1])
                        cc1.caption(f"{truncate_uuid(cl_id)} · {when} · {item_count} items")
                        if cc2.button("View", key=f"view_{cl_id}", use_container_width=True):
                            st.query_params.update(page="viewer", checklist_id=cl_id)
                            st.rerun()
            except Exception:
                pass

            st.markdown("</div>", unsafe_allow_html=True)

            if st.button(
                "Generate for this case",
                key=f"gen_{cid}",
                use_container_width=True,
            ):
                st.session_state["active_case_id"] = cid
                st.rerun()


def render() -> None:
    """Render the full Case Overview page."""
    _generate_form()
    st.divider()
    _upload_card()
    st.divider()
    _cases_grid()
