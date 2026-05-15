"""Design tokens, CSS palettes, and theme-injection helpers.

Two named themes:
  - parchment  warm off-white + terracotta accent (light default)
  - ink        deep midnight + brightened accents (dark)
  - system     auto via @media (prefers-color-scheme: dark)

All CSS components reference only semantic tokens (--bg, --accent, etc.),
never raw palette colours, so swapping themes costs zero per-component work.
"""
from __future__ import annotations

import streamlit as st

# ─── Palette dictionaries ────────────────────────────────────────────────────

PARCHMENT: dict[str, str] = {
    "--bg": "#f7f2e7",
    "--surface": "#fbf7ec",
    "--surface-raised": "#ffffff",
    "--border": "#ece6d7",
    "--text": "#1a1d2e",
    "--text-muted": "#6b6d7e",
    "--accent": "#d97757",
    "--accent-faint": "rgba(217,119,87,0.12)",
    "--learning": "#5b6dcd",
    "--learning-faint": "rgba(91,109,205,0.12)",
    "--status-present": "#4a7c59",
    "--status-present-bg": "rgba(74,124,89,0.12)",
    "--status-unclear": "#c89b3c",
    "--status-unclear-bg": "rgba(200,155,60,0.12)",
    "--status-missing": "#8a8a8a",
    "--status-missing-bg": "rgba(138,138,138,0.12)",
    "--shadow": "rgba(26,29,46,0.08)",
    "--shadow-raised": "rgba(26,29,46,0.14)",
}

INK: dict[str, str] = {
    "--bg": "#14162a",
    "--surface": "#1d2040",
    "--surface-raised": "#262a55",
    "--border": "#2a2e52",
    "--text": "#f0ead9",
    "--text-muted": "#9b9aab",
    "--accent": "#e88968",
    "--accent-faint": "rgba(232,137,104,0.15)",
    "--learning": "#8d9dee",
    "--learning-faint": "rgba(141,157,238,0.15)",
    "--status-present": "#6aa17a",
    "--status-present-bg": "rgba(106,161,122,0.15)",
    "--status-unclear": "#e0b760",
    "--status-unclear-bg": "rgba(224,183,96,0.15)",
    "--status-missing": "#a8a8a8",
    "--status-missing-bg": "rgba(168,168,168,0.15)",
    "--shadow": "rgba(0,0,0,0.35)",
    "--shadow-raised": "rgba(0,0,0,0.5)",
}

# ─── Semantic constants for Python code ──────────────────────────────────────

CATEGORY_ORDER: tuple[str, ...] = (
    "Parties",
    "Financial Terms",
    "Required Exhibits",
    "Signatures",
    "Deadlines",
    "Consents",
    "Disclosures",
    "Other",
)

PATTERN_TYPES: tuple[str, ...] = (
    "rename_rule",
    "template_addition",
    "template_removal",
    "status_default",
    "style_preference",
    "category_remap",
)

STATUS_STYLES: dict[str, dict[str, str]] = {
    "present": {
        "color": "var(--status-present)",
        "bg": "var(--status-present-bg)",
        "glyph": "●",
        "label": "Present",
        "cls": "lex-badge-present",
    },
    "unclear": {
        "color": "var(--status-unclear)",
        "bg": "var(--status-unclear-bg)",
        "glyph": "◐",
        "label": "Unclear",
        "cls": "lex-badge-unclear",
    },
    "missing": {
        "color": "var(--status-missing)",
        "bg": "var(--status-missing-bg)",
        "glyph": "○",
        "label": "Missing",
        "cls": "lex-badge-missing",
    },
}

# ─── Component CSS (palette-agnostic — uses only var(...)) ───────────────────

_COMPONENT_CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');

/* ── Base resets ── */
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stAppViewContainer"] > .main > .block-container { background: var(--bg) !important; }
[data-testid="stSidebar"] > div:first-child {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stHeader"] {
  background: var(--bg) !important;
  border-bottom: 1px solid var(--border) !important;
}
.stApp { background: var(--bg) !important; }
.block-container { max-width: 1100px !important; padding-top: 1.5rem !important; }

/* ── Typography overrides ── */
body, .stMarkdown p, label, .stText { color: var(--text) !important; }
h1, h2, h3, h4, h5, h6 {
  color: var(--text) !important;
  font-family: 'Fraunces', Georgia, serif !important;
}
.stCaption p { color: var(--text-muted) !important; }

/* ── Lexicon banner ── */
.lexicon-banner {
  background: linear-gradient(135deg, #1a1d2e 0%, #2a2f52 100%);
  border-radius: 14px;
  padding: 1.5rem 2rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.18);
  border: 1px solid rgba(255,255,255,0.06);
}
.lexicon-banner .lex-wordmark {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 2.1rem;
  font-weight: 700;
  color: #f0ead9;
  margin: 0;
  letter-spacing: -0.03em;
  line-height: 1.1;
}
.lexicon-banner .lex-wordmark .lex-accent {
  color: var(--accent);
  border-bottom: 2.5px solid var(--accent);
  padding-bottom: 1px;
}
.lexicon-banner .lex-tagline {
  color: #9b9aab;
  margin: 0.3rem 0 0;
  font-size: 0.875rem;
  font-family: 'Inter', sans-serif;
}

/* ── Item & case cards ── */
.lex-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 1px 4px var(--shadow);
  transition: box-shadow 160ms ease-out, border-color 160ms ease-out;
}
.lex-card:hover {
  box-shadow: 0 6px 18px var(--shadow-raised);
  border-color: var(--accent);
}

/* ── Status & generic badges ── */
.lex-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.3em;
  padding: 0.18em 0.65em;
  border-radius: 20px;
  font-size: 0.76rem;
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  letter-spacing: 0.01em;
  white-space: nowrap;
}
.lex-badge-present { color: var(--status-present); background: var(--status-present-bg); }
.lex-badge-unclear { color: var(--status-unclear); background: var(--status-unclear-bg); }
.lex-badge-missing { color: var(--status-missing); background: var(--status-missing-bg); }
.lex-badge-learning { color: var(--learning); background: var(--learning-faint); }
.lex-badge-action  { color: var(--accent);   background: var(--accent-faint);   }
.lex-badge-neutral {
  color: var(--text-muted); background: var(--surface);
  border: 1px solid var(--border);
}

/* ── Confidence bar ── */
.lex-conf-wrap { display: inline-flex; align-items: center; gap: 0.4rem; }
.lex-conf-track {
  display: inline-block; width: 72px; height: 5px;
  background: var(--border); border-radius: 3px; vertical-align: middle;
}
.lex-conf-fill {
  display: block; height: 100%; border-radius: 3px;
  background: var(--accent);
}
.lex-conf-label {
  font-size: 0.76rem; color: var(--text-muted);
  font-family: 'Inter', sans-serif;
}

/* ── Stats strip ── */
.lex-stats {
  display: flex;
  gap: 1.5rem;
  align-items: flex-end;
  padding: 0.7rem 1rem;
  background: var(--surface);
  border-radius: 8px;
  border: 1px solid var(--border);
  margin-bottom: 1rem;
  flex-wrap: wrap;
  font-family: 'Inter', sans-serif;
  font-size: 0.83rem;
  color: var(--text-muted);
}
.lex-stats .stat-val { font-weight: 600; }
.lex-hist-bar {
  display: inline-block;
  width: 7px;
  border-radius: 2px 2px 0 0;
  background: var(--accent);
  margin-right: 2px;
  vertical-align: bottom;
  opacity: 0.75;
}

/* ── Evidence chip (expander label override) ── */
.lex-evid {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.15em 0.5em;
  color: var(--text-muted);
}

/* ── Skeleton shimmer ── */
@keyframes lex-shimmer {
  0%   { background-position: -600px 0; }
  100% { background-position:  600px 0; }
}
.lex-skeleton {
  background: linear-gradient(
    90deg,
    var(--surface) 25%,
    var(--border)  50%,
    var(--surface) 75%
  );
  background-size: 1200px 100%;
  animation: lex-shimmer 1.6s ease-in-out infinite;
  border-radius: 8px;
}

/* ── Sidebar radio as pills ── */
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label {
  border-radius: 8px !important;
  padding: 0.4rem 0.85rem !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
  color: var(--text) !important;
  transition: background 140ms ease, color 140ms ease !important;
}
/* Target inner text nodes explicitly — Streamlit nests text inside p/span inside label */
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio label span:not([data-testid]),
[data-testid="stSidebar"] .stRadio label div:not(:first-child) {
  color: var(--text) !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
  background: var(--accent-faint) !important;
  color: var(--accent) !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p,
[data-testid="stSidebar"] .stRadio label:has(input:checked) span:not([data-testid]),
[data-testid="stSidebar"] .stRadio label:has(input:checked) div:not(:first-child) {
  color: var(--accent) !important;
}
[data-testid="stSidebar"] .stRadio label:hover { background: var(--border) !important; }

/* ── Theme-swap smooth transition ── */
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="stHeader"],
.lex-card, .lex-badge, .lex-stats, .lexicon-banner {
  transition: background-color 220ms ease, border-color 220ms ease, color 220ms ease !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Buttons — override Streamlit's default dark style ── */
.stButton > button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  background: var(--surface-raised) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 2px 8px var(--shadow) !important;
}

/* ── Input fields ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: var(--surface-raised) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}
/* Selectbox */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > div > div > div {
  background: var(--surface-raised) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}
/* File uploader */
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text-muted) !important;
}
[data-testid="stFileUploader"] button {
  background: var(--surface-raised) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}
/* Expander */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border-color: var(--border) !important;
}
/* Popover / dialog */
[data-testid="stPopover"] > div,
.stPopover { background: var(--surface-raised) !important; border-color: var(--border) !important; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
  .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
  .lexicon-banner { padding: 1rem 1.25rem; }
  .lexicon-banner .lex-wordmark { font-size: 1.6rem; }
  .lex-stats { gap: 0.75rem; font-size: 0.78rem; }
}
"""


def _build_root_vars(palette: dict[str, str]) -> str:
    lines = "\n".join(f"  {k}: {v};" for k, v in palette.items())
    return f":root {{\n{lines}\n}}"


def build_css(theme: str) -> str:
    """Assemble the full CSS block for the given theme choice."""
    if theme == "system":
        parchment_vars = _build_root_vars(PARCHMENT)
        ink_lines = "\n".join(f"    {k}: {v};" for k, v in INK.items())
        system_override = (
            f"@media (prefers-color-scheme: dark) {{\n  :root {{\n{ink_lines}\n  }}\n}}"
        )
        return f"{parchment_vars}\n{system_override}\n{_COMPONENT_CSS}"
    palette = INK if theme == "ink" else PARCHMENT
    return f"{_build_root_vars(palette)}\n{_COMPONENT_CSS}"


def render_theme(theme: str) -> None:
    """Inject the CSS block for the active theme into the Streamlit page."""
    st.markdown(f"<style>{build_css(theme)}</style>", unsafe_allow_html=True)
