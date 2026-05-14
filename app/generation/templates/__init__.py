"""Template registry — maps template slug to ChecklistTemplate (PRD §5e)."""

from app.generation.templates.commercial_contract import COMMERCIAL_CONTRACT
from app.generation.templates.nda import NDA
from app.generation.templates.types import ChecklistTemplate, TemplateItem

TEMPLATES: dict[str, ChecklistTemplate] = {
    "commercial_contract": COMMERCIAL_CONTRACT,
    "nda": NDA,
}

__all__ = ["TEMPLATES", "ChecklistTemplate", "TemplateItem"]
