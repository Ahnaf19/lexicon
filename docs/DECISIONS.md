# Design Decisions

This document explains the six non-obvious architectural choices in Lexicon and what was rejected in each case.

---

## 1. Grounding is enforced by invariant, not by prompt

**Decision:** Every item with `status="present"` is required by the Pydantic model to have at least one non-empty `evidence` citation. This constraint is checked at the schema boundary, not by a downstream policy.

**Why:** Prompting the LLM to "only say present if you have evidence" is unreliable — the model may produce a confident rationale with no citation, or fabricate a page number. The V1–V5 validation node coerces ambiguous items to `"unclear"` and drops any citation whose page number doesn't exist in the ingested document. The SQL invariant `COUNT(*) WHERE status='present' AND evidence=∅` returns zero rows across every generation run — it's a measurable hard constraint, not a policy.

**Rejected:** Downstream filtering (drop items with low confidence). This still allows hallucinated cites to reach the user, just at lower confidence. Unclear is a first-class status for a reason — it is more honest than a wrong answer.

> [!WARNING]
> If you bypass the validation node (e.g. by calling `assemble` directly), the Pydantic model will still reject `status="present"` + empty evidence at the API boundary. The constraint is in two places intentionally.

---

## 2. Parent-section expansion for the chunking-vs-context tension

**Decision:** Retrieval ranks at the **512-token window** level for precision; generation receives the **full parent section** (up to 3,500 tokens) for context fidelity. Citations anchor to the precise window span.

**Why:** If you chunk at 3,500 tokens, precision suffers — every query retrieves walls of text and the relevant sentence ranks poorly. If you chunk at 512 tokens, context suffers — the LLM sees only a sentence fragment and misses the clause that qualifies it. Parent expansion gives you both: the dense index finds the right sentence, then the model reads the full clause around it.

**Rejected:** Re-ranking with a cross-encoder. A cross-encoder on a 10–20 document corpus with 271 windows adds latency with marginal precision gain at this scale. It's in the deferred list and documented in [ROADMAP.md](./ROADMAP.md).

---

## 3. Groq primary + Ollama fallback, not local-first

**Decision:** `LLM_PROVIDER=groq` is the default. The same `init_chat_model` abstraction switches to `LLM_PROVIDER=ollama` without code changes.

**Why:** A reviewer without an NVIDIA GPU would wait 8–15 minutes per checklist run with a local 8B model. Groq runs `llama-3.3-70b-versatile` at ~400 tokens/sec on custom LPU hardware — a full 12-item checklist completes in 60–110 seconds on the free tier. Groq's free tier covers the entire demo comfortably. The local option exists for users who can't or won't share an API key.

**Rejected:** LiteLLM as the provider abstraction. `init_chat_model` (LangChain) is already a dependency and covers the two providers Lexicon needs. Adding LiteLLM would be a second abstraction layer over the same thing.

---

## 4. OCR triage, not a single engine

**Decision:** Marker (Surya) is the primary OCR engine. Blocks with `confidence < 0.6` fall back to `microsoft/trocr-large-handwritten`.

**Why:** Marker handles native PDFs, scanned PDFs, and clean images in one code path. It auto-detects extractable text vs. image-rendered pages and preserves layout (columns, tables, reading order) — important for legal documents with exhibit schedules. TrOCR is only loaded when a block fails the confidence gate; loading it for every page would be slow and unnecessary for clean contracts.

**Rejected:** Tesseract as the primary engine. Marker outperforms Tesseract on the olmOCR-Bench benchmark by a wide margin, particularly on multi-column layouts and low-resolution scans. The handwritten exhibit came through as "code for the patforn" with TrOCR on a noisy scan — Tesseract would have done worse. The bottleneck is scan quality, not the model.

**Also rejected:** VLM-based OCR (Chandra, DeepSeek-OCR, Nanonets OCR 2). These 4–9B models compete with the generation LLM for GPU and are 4–12× slower on CPU. They're in the deferred list for a future release where GPU allocation can be planned.

---

## 5. Corroboration gate of ≥ 3 edits for pattern promotion

**Decision:** A `LearnedPattern` is only promoted to the active generation prompt when at least 3 different operator edits corroborate it.

**Why:** A single edit could be a mistake, a document-specific quirk, or an operator who changed their mind. Three independently corroborating edits establish that the pattern reflects a stable preference, not noise. The confidence formula compounds edit count and recency so that older weak patterns decay relative to fresh strong ones.

**Rejected:** Promote on first edit. This is maximally responsive but maximally fragile — a stray edit promotes bad rules into every subsequent generation. The goal of the loop is to make future runs *better*, not just *different*.

**Rejected:** Require 5+ edits. This is too conservative for a demo with limited edit volume. Three is the minimum that distinguishes signal from coincidence while still being reachable in a short demo session.

---

## 6. Real Postgres in every test, not mock sessions

**Decision:** Integration tests run against a `lexicon_test` database with SAVEPOINT-per-test isolation. No mocked sessions anywhere in the test suite.

**Why:** The last time this project used mocked DB sessions (phase 2 of development), a Pydantic schema change broke the ORM layer silently — the mocked test passed, production insert failed. A SAVEPOINT fixture is only marginally slower than a mock for tests that don't need network I/O, and it catches real constraint violations, schema drift, and `asyncpg` driver quirks that mocks cannot surface.

**Rejected:** SQLite for tests. SQLite doesn't support pgvector, ARRAY columns, or most of the Postgres-specific query features Lexicon uses. Every test that touches retrieval would need a shim.
