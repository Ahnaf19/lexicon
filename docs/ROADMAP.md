# Roadmap

Known limitations of the current implementation and what would come next in a production build.

---

## Known limitations

### TrOCR quality on degraded handwriting

The handwritten exhibit used in the demo is a noisy photo scan. TrOCR produces legible but imperfect output — "code for the platform" came through as "code for the patforn". A cleaner scan or a higher-resolution capture would materially improve recall on handwriting-specific queries. The retrieval logic is sound; the bottleneck is image quality, not the model.

Dense embeddings are robust to moderate OCR noise (the handwritten exhibit still ranked #2 on semantic queries despite the garbled text), but exact-phrase searches and sparse BM25 scores degrade proportionally to OCR error rate.

### Pattern extraction requires edit volume

The ≥ 3 corroboration gate is intentionally conservative. With the demo documents and a single edit per run, patterns promote slowly (one pattern after 3 runs). In production with real operator traffic — dozens of edits per matter, hundreds of matters per month — the loop would fill quickly and cover many more item types. The gate trades responsiveness for reliability; it is parameterised and easy to adjust.

### Operator identity is wired but unauthenticated

Every `EditEvent` row carries an `actor` field populated from the `X-Operator-Id` request header. The surface for a real IdP is in place — binding it to Auth0, Cognito, or any FastAPI `Depends` provider is a configuration change, not an architectural one. For the demo, all edits default to the actor `"default"`.

### Groq free-tier throughput

At ~100K tokens/day, a back-to-back 4-run eval can exhaust the quota mid-session. The eval harness resets learning state per session so iteration is cheap once quota refreshes. Production would use a paid tier or a self-hosted endpoint.

### Sequential item processing

LangGraph nodes process checklist items one at a time (controlled by `MAX_PARALLEL_ITEMS = 1` in `app/generation/graph.py`). Groq's ~400 TPS throughput means a 12-item checklist completes in ~60–110 seconds. Raising `MAX_PARALLEL_ITEMS` to 4–6 would reduce wall-clock time to ~15–25 seconds with no code changes to the node logic — it is deliberately deferred because Ollama on shared memory (MPS) is not safely parallelisable.

---

## What I'd ship next

Listed in priority order.

### 1. Operator authentication (1 day)

Replace `X-Operator-Id` header with a real JWT or session cookie. Any FastAPI `Depends` auth provider works. This is the minimum for a real multi-user deployment.

### 2. Groq paid tier + parallel items (half day)

Raise `MAX_PARALLEL_ITEMS` in `graph.py` and switch to a paid Groq key. Reduces generation time from ~90 s to ~20 s with no other changes.

### 3. Cross-encoder reranker (2 days)

After hybrid RRF, pass the top-20 chunks through a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker before parent expansion. Expected improvement: +5–8% citation precision on ambiguous queries (where dense and sparse disagree on the top result). Deferred because 271 windows across 8 documents is too small a corpus to demonstrate the gain convincingly.

### 4. Edit-history panel in the UI (1 day)

Add a `GET /checklists/{id}/edits` endpoint (simple `SELECT * FROM edit_events WHERE checklist_id=?`) and expose it as a collapsible "Edit history" panel per item in the Viewer page. All the data already exists — it just needs a UI surface.

### 5. Template registry expansion (ongoing)

The template registry ships with `commercial_contract` and `nda`. Each new doc type (lease, employment agreement, IP assignment) is a JSON template file and a few lines of registration code. No schema changes.

### 6. Qdrant migration (1 week)

Replace pgvector with Qdrant for vector storage. Benefits: native sparse + dense hybrid indexes, built-in metadata filtering without a JOIN, payload-indexed filtering. Cost: removes the "one database" simplicity. Postgres still owns all relational state; Qdrant handles vectors only. Deferred until the corpus outgrows pgvector's HNSW performance envelope (roughly 10M+ vectors).

### 7. DSPy / TextGrad automatic prompt optimisation (2 weeks)

Replace hand-written few-shot prompts in `draft_item` and `critique` with DSPy-compiled prompts optimised against the `mean_edit_distance` metric. The `LearnedPattern` infrastructure already provides the training signal — DSPy would consume it automatically rather than requiring a human to write extraction rules.

### 8. Multi-language support (4 weeks)

The current pipeline is English-only. `nomic-embed-text` handles multilingual input reasonably well at the retrieval layer. The main work is template localisation and training/testing the LLM with non-English legal document structures.

### 9. Fine-tuning / LoRA

Deferred indefinitely for this scope. The few-shot + pattern approach provides a measurable improvement loop without the data and infrastructure overhead of fine-tuning. The architecture is compatible with a fine-tuned base model — swapping in a LoRA adapter is a provider config change — but the PoC doesn't need it.
