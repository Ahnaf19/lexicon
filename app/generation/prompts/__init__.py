"""Prompt loader — reads versioned prompt templates from the package data."""

from __future__ import annotations

import importlib.resources


def load_prompt(name: str, version: str = "v1") -> str:
    """Load a prompt template by name and version string."""
    pkg = importlib.resources.files(f"app.generation.prompts.{version}")
    return (pkg / f"{name}.txt").read_text(encoding="utf-8")
