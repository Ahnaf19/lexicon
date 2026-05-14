"""Tests that both templates load correctly with expected item counts and slugs."""

from __future__ import annotations

import pytest

from app.generation.templates import TEMPLATES


def test_all_templates_present():
    assert "commercial_contract" in TEMPLATES
    assert "nda" in TEMPLATES


def test_commercial_contract_items():
    tmpl = TEMPLATES["commercial_contract"]
    assert len(tmpl.items) == 12
    slugs = {item.slug for item in tmpl.items}
    expected = {
        "parties", "effective_date", "term_duration", "termination",
        "confidentiality", "ip_ownership", "indemnification", "liability_limit",
        "governing_law", "notice_provisions", "assignment", "signatures",
    }
    assert slugs == expected


def test_nda_items():
    tmpl = TEMPLATES["nda"]
    assert len(tmpl.items) == 10
    slugs = {item.slug for item in tmpl.items}
    expected = {
        "parties", "effective_date", "definition_confidential", "permitted_use",
        "exclusions", "term_survival", "return_destruction", "remedies",
        "governing_law", "signatures",
    }
    assert slugs == expected


def test_template_ids_deterministic():
    t1 = TEMPLATES["commercial_contract"]
    t2 = TEMPLATES["commercial_contract"]
    assert t1.id == t2.id


def test_template_doc_types():
    assert TEMPLATES["commercial_contract"].doc_type == "commercial_contract"
    assert TEMPLATES["nda"].doc_type == "nda"


def test_items_have_sub_queries():
    for slug, tmpl in TEMPLATES.items():
        for item in tmpl.items:
            assert item.sub_query, f"Template {slug} item {item.slug} missing sub_query"


def test_stable_uuid_deterministic():
    tmpl = TEMPLATES["nda"]
    item = tmpl.items[0]
    u1 = item.stable_uuid(tmpl.id)
    u2 = item.stable_uuid(tmpl.id)
    assert u1 == u2


def test_stable_uuid_distinct_for_different_slugs():
    tmpl = TEMPLATES["commercial_contract"]
    uuids = [item.stable_uuid(tmpl.id) for item in tmpl.items]
    assert len(uuids) == len(set(uuids)), "stable_uuid collisions across template items"
