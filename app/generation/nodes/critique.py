"""critique node — pass-through stub; phase 5 wires learned_patterns here."""

from __future__ import annotations

from app.generation.state import ChecklistState


async def critique(state: ChecklistState) -> dict[str, object]:
    """Apply learned patterns (phase 5). Currently a pass-through."""
    return {}
