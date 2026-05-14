"""LangGraph pipeline for checklist generation (PRD §5e).

Graph topology:
  START → warmup_node → classify_doc_set → load_template → router
  router → retrieve_evidence → draft_item → validate_item → critique → router
  router (exhausted) → assemble → END

Items are processed SEQUENTIALLY (G1). The router advances _item_index each pass.
Phase 6 (Groq) can lift MAX_PARALLEL_ITEMS without touching node code.
"""

from __future__ import annotations


from langgraph.graph import END, START, StateGraph
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from app.generation.nodes.assemble import assemble
from app.generation.nodes.classify_doc_set import classify_doc_set
from app.generation.nodes.critique import critique
from app.generation.nodes.draft_item import draft_item
from app.generation.nodes.load_template import load_template
from app.generation.nodes.retrieve_evidence import retrieve_evidence
from app.generation.nodes.validate_item import validate_item
from app.generation.state import ChecklistState
from app.generation.warmup import warmup_llm

# Phase 6: raise to enable Groq fan-out. Sequential for now (Ollama + MPS safety).
MAX_PARALLEL_ITEMS = 1


# ---------------------------------------------------------------------------
# warmup wrapper (LangGraph node signature)
# ---------------------------------------------------------------------------


async def warmup_node(state: ChecklistState) -> dict[str, object]:
    await warmup_llm()
    return {}


# ---------------------------------------------------------------------------
# Per-item loop router
# ---------------------------------------------------------------------------


def _advance_item(state: ChecklistState) -> dict[str, object]:
    """Advance the item index cursor; set current_item_slug for the next node."""
    template = state["template"]
    idx: int = state.get("item_index", 0) + 1
    if idx < len(template.items):
        return {
            "item_index": idx,
            "current_item_slug": template.items[idx].slug,
        }
    return {"item_index": idx, "current_item_slug": None}


async def router_node(state: ChecklistState) -> dict[str, object]:
    """Advance the item cursor; used both as an entry and as the post-critique redirect."""
    return _advance_item(state)


def _route_after_router(state: ChecklistState) -> str:
    """Conditional edge: go to retrieve_evidence if items remain, else assemble."""
    if state.get("current_item_slug") is not None:
        return "retrieve_evidence"
    return "assemble"


def _route_after_load_template(state: ChecklistState) -> str:
    """After load_template, start the first item directly (index already = 0)."""
    template = state.get("template")
    if template and template.items:
        return "retrieve_evidence"
    return "assemble"


# ---------------------------------------------------------------------------
# Graph build
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph[ChecklistState]:
    g: StateGraph[ChecklistState] = StateGraph(ChecklistState)

    g.add_node("warmup_node", warmup_node)
    g.add_node("classify_doc_set", classify_doc_set)
    g.add_node("load_template", load_template)
    g.add_node("retrieve_evidence", retrieve_evidence)
    g.add_node("draft_item", draft_item)
    g.add_node("validate_item", validate_item)
    g.add_node("critique", critique)
    g.add_node("router_node", router_node)
    g.add_node("assemble", assemble)

    g.add_edge(START, "warmup_node")
    g.add_edge("warmup_node", "classify_doc_set")
    g.add_edge("classify_doc_set", "load_template")
    g.add_conditional_edges(
        "load_template",
        _route_after_load_template,
        {"retrieve_evidence": "retrieve_evidence", "assemble": "assemble"},
    )
    g.add_edge("retrieve_evidence", "draft_item")
    g.add_edge("draft_item", "validate_item")
    g.add_edge("validate_item", "critique")
    g.add_edge("critique", "router_node")
    g.add_conditional_edges(
        "router_node",
        _route_after_router,
        {"retrieve_evidence": "retrieve_evidence", "assemble": "assemble"},
    )
    g.add_edge("assemble", END)

    return g


# ---------------------------------------------------------------------------
# Compiled graph singleton
# ---------------------------------------------------------------------------

_compiled = None


def get_compiled_graph() -> CompiledStateGraph[Any, Any, Any]:
    """Return the compiled graph, initialising once per process."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()
    return _compiled
