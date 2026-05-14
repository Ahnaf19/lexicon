"""Section detection regex coverage — markdown, all-caps, numbered legal headings."""

from __future__ import annotations

import pytest

from app.retrieval.chunking import _find_section_boundaries


def _headings_at(text: str) -> list[str]:
    """Return the first lines at each detected boundary (except 0 if implicit)."""
    boundaries = _find_section_boundaries(text)
    result = []
    for b in boundaries:
        first_line = text[b:].split("\n", 1)[0].strip()
        result.append(first_line)
    return result


@pytest.mark.parametrize(
    "text,expected_heading",
    [
        # Markdown headings
        ("# Introduction\nSome text here.", "# Introduction"),
        ("## Definitions\nAs used herein.", "## Definitions"),
        ("### 3.1 Scope\nLimited to.", "### 3.1 Scope"),
        # All-caps section lines
        ("CONFIDENTIALITY\nAll information shared.", "CONFIDENTIALITY"),
        ("GOVERNING LAW AND JURISDICTION\nThis agreement.", "GOVERNING LAW AND JURISDICTION"),
        # ARTICLE / SECTION headings
        ("ARTICLE I\nGeneral provisions apply.", "ARTICLE I"),
        ("SECTION IV Definitions\nFor purposes of.", "SECTION IV Definitions"),
        ("ARTICLE 12\nMiscellaneous terms.", "ARTICLE 12"),
        # Numbered legal headings
        ("1.1 Definitions\nThe following terms.", "1.1 Definitions"),
        ("2.3.1 Termination\nEither party may.", "2.3.1 Termination"),
    ],
)
def test_heading_detected(text: str, expected_heading: str) -> None:
    headings = _headings_at(text)
    assert any(expected_heading in h for h in headings), (
        f"Expected heading '{expected_heading}' not detected in {headings!r}"
    )


def test_multi_section_boundaries() -> None:
    text = (
        "# Introduction\nPreamble text.\n\n"
        "## Definitions\nDefined terms.\n\n"
        "ARTICLE I\nGeneral provisions."
    )
    boundaries = _find_section_boundaries(text)
    assert len(boundaries) >= 3  # 0 + at least 2 detected headings


def test_plain_prose_no_extra_boundaries() -> None:
    text = "This agreement is entered into by the parties.\n" * 10
    boundaries = _find_section_boundaries(text)
    # Only the implicit 0 offset; no false positives from prose
    assert boundaries == [0]


def test_boundaries_sorted_and_unique() -> None:
    text = (
        "# Section One\nContent.\n"
        "## Section Two\nMore content.\n"
        "ARTICLE I\nProvisions.\n"
    )
    boundaries = _find_section_boundaries(text)
    assert boundaries == sorted(set(boundaries))
