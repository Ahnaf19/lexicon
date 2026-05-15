"""Shared utility helpers — no external deps beyond stdlib."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterator


def humanize_relative_time(dt: datetime) -> str:
    """Return a human-readable relative time string like '3h ago'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        m = seconds // 60
        return f"{m}m ago"
    if seconds < 86400:
        h = seconds // 3600
        return f"{h}h ago"
    d = seconds // 86400
    return f"{d}d ago"


def truncate_uuid(u: object) -> str:
    """Return first 8 chars of a UUID string for compact display."""
    return str(u)[:8] + "…"


def sse_iter_events(lines: Iterator[str]) -> Iterator[tuple[str, dict]]:  # type: ignore[type-arg]
    """Parse an SSE line stream into (event_name, data_dict) tuples.

    Handles the 'event:' / 'data:' two-line SSE format emitted by
    POST /checklists/generate.
    """
    current_event = "message"
    for line in lines:
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            try:
                yield current_event, json.loads(data_str)
            except json.JSONDecodeError:
                pass
            current_event = "message"  # reset after dispatch per SSE spec §9.2.6


def parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp string, returning None on failure."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
