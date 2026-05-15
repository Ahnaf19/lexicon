# Lexicon

Lexicon ingests messy legal documents — scanned contracts, degraded PDFs, handwritten exhibits — and produces structured Document Checklists where every claim cites a specific span in the source material.

Unlike naive RAG pipelines that generate plausible-sounding answers, Lexicon enforces grounding: a V1–V5 validation gate rejects hallucinated page references and coerces ambiguous evidence to `unclear` rather than guessing. Parent-section expansion ensures citations still point at precise text windows while generation sees the full surrounding context.

The improvement loop is operational — operator edits become typed database rows, distilled into promoted `LearnedPattern` rules, and applied automatically on the next run.

---

## Quickstart

### Prerequisites

| Tool | Install |
|---|---|
| Docker Desktop | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| uv (Python package manager) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (Mac/Linux) or `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` (Windows) |
| Groq API key (free) | [console.groq.com](https://console.groq.com) — create an account, copy your key |

### 1. Clone and configure

```bash
git clone <repo-url>
cd legal_ai

cp .env.template .env
# Open .env and set GROQ_API_KEY=<your-key>
```

### 2. Start Postgres and sync Python deps

```bash
docker compose up -d postgres
uv sync
```

### 3. Run migrations and ingest sample documents

```bash
uv run alembic upgrade head
uv run python -m app.cli ingest samples/clean samples/degraded samples/handwritten
```

### 4. Start the API

```bash
uv run uvicorn app.main:app --reload
# API is live at http://localhost:8000
# OpenAPI docs at http://localhost:8000/docs
```

### 5. Open the UI

```bash
uv run streamlit run ui/streamlit_app.py
# UI opens at http://localhost:8501
```

Or use `make`:

```bash
make api   # starts uvicorn with --reload
make ui    # starts Streamlit
```

### Generate a checklist via CLI

```bash
uv run python -m app.cli checklist generate \
  --case-id 00000000-0000-0000-0000-000000000001 \
  --template commercial_contract
```

---

## Windows quickstart (WSL2 + Docker Desktop)

Docker Desktop on Windows uses WSL2 as its backend — all commands run the same as on Mac/Linux once Docker is running.

1. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) with the WSL2 backend enabled.
2. Install uv in PowerShell: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
3. Open a WSL2 terminal (or use the Docker Desktop terminal) and follow the Quickstart steps above verbatim.

### NVIDIA GPU support (RTX 3060 Ti and other CUDA cards)

Ollama (used for local embeddings and the Ollama LLM fallback) can use your GPU for faster inference.

**Setup:**

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your WSL2 host:
   ```bash
   # In WSL2 Ubuntu:
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
2. Verify: `docker run --gpus all nvidia/cuda:12.0-base nvidia-smi` — should print your GPU name.

3. Start Ollama with GPU access:
   ```bash
   docker compose --profile local-llm up -d ollama
   ```
   The `deploy.resources.reservations.devices` block in `docker-compose.yml` is already wired — no further changes needed.

4. Pull the embedding model:
   ```bash
   docker compose exec ollama ollama pull nomic-embed-text
   # For local LLM fallback:
   docker compose exec ollama ollama pull qwen3:8b
   ```

5. Set `LLM_PROVIDER=ollama` in `.env` to switch from Groq to the local GPU.

**Verifying GPU is in use:**

Ollama logs GPU memory allocation when a model is loaded:
```
# docker compose logs -f ollama
time=... msg="loading model" gpu=0 name=NVIDIA GeForce RTX 3060 Ti vram=7.67GiB
time=... msg="llama runner started" layers_on_gpu=33
```

You can also run `nvidia-smi` in WSL2 while a generation is in progress — look for the `ollama` process in the GPU Process column.

**Performance: RTX 3060 Ti vs Apple M3 Pro**

| Task | M3 Pro (18 GB unified) | RTX 3060 Ti (8 GB VRAM) |
|---|---|---|
| `nomic-embed-text` inference | ~180 tokens/s | ~400–600 tokens/s |
| `qwen3:8b` generation | ~30–40 tokens/s | ~55–80 tokens/s |
| Marker OCR (clean PDF, 10 pages) | ~8–12 s | ~2–4 s (GPU path) |

The RTX 3060 Ti's dedicated VRAM gives it a clear edge on transformer inference — expect roughly 2× faster embedding throughput and 1.5–2× faster local LLM generation compared to the M3 Pro's shared-memory GPU. The M3 Pro's advantage is zero model-load overhead (no PCIe transfer) and higher sustained CPU bandwidth for preprocessing.

> **Note:** Groq (`LLM_PROVIDER=groq`) runs on Groq's custom LPU hardware at ~400 tokens/s regardless of local GPU. For generation, Groq is faster than both options above. GPU acceleration primarily benefits local embeddings (Ollama `nomic-embed-text`) and local fallback LLM (`qwen3:8b`).

---

## Running with full Docker stack

Bring up every service with one command:

```bash
docker compose up -d                          # postgres + api + streamlit
docker compose --profile local-llm up -d     # + ollama (for local embeddings/LLM)
docker compose --profile obs up -d           # + Langfuse tracing UI (port 3000)
```

Then run migrations once:
```bash
docker compose exec api uv run alembic upgrade head
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION                                                       │
│  PDFs / JPGs ──▶ Marker (Surya OCR) ──▶ blocks                  │
│                  ├─ confidence < 0.6 ──▶ TrOCR fallback         │
│                  └─ structured extraction (LLM)                  │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  RETRIEVAL                                                       │
│  Blocks ──▶ section-aware chunking ──▶ windows + parent sections │
│             ──▶ nomic-embed-text (768d, HNSW cosine)             │
│             ──▶ tsvector (BM25 via ts_rank_cd)                   │
│  Query ──▶ dense ‖ sparse ──▶ RRF (k=60) ──▶ parent expansion   │
│             ──▶ SearchHit (with context_text)                    │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  GENERATION (LangGraph)                                          │
│  classify ▶ load_template ▶ FOR EACH ITEM:                       │
│      retrieve_evidence ▶ draft_item ▶ validate (V1–V5) ▶ critique│
│  ▶ assemble                                                      │
│  Output: Checklist with grounded items + EvidenceCitations       │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LEARNING (edit loop)                                            │
│  Operator PATCH/POST/DELETE ──▶ typed EditEvent rows             │
│  POST /finalize ──▶ pattern_extractor (1 LLM call)               │
│      ──▶ LearnedPattern (promoted at corroboration≥3, conf≥0.7)  │
│      ──▶ few_shot_examples (cosine-retrieved at draft time)      │
│  Next generation: load_template applies template mutations,      │
│                   critique applies rename/style/status rules     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Example output

A single generated checklist item, showing full grounding (UUIDs omitted):

```json
{
  "title": "Term and duration",
  "category": "Deadlines",
  "status": "present",
  "required": true,
  "confidence": 0.94,
  "rationale": "Section 11.1 specifies an initial term of one year commencing on
    the Commencement Date, with automatic renewal unless either party gives 30 days
    written notice prior to renewal.",
  "evidence": [
    {
      "page_number": 10,
      "char_offset_start": 4821,
      "char_offset_end": 4973,
      "snippet": "11.1 This agreement begins on the Commencement Date and continues
        for an initial term of one (1) year, renewing automatically unless
        terminated by written notice no fewer than thirty (30) days prior
        to renewal.",
      "retrieval_score": 0.91
    }
  ],
  "learned_from_pattern_ids": []
}
```

The validator enforces that `evidence` is non-empty whenever `status` is `"present"` — the constraint is in the Pydantic model, not downstream logic.

---

## API reference

The OpenAPI spec is at `http://localhost:8000/docs` when the server is running. Key endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/healthz` | DB reachability check |
| `POST` | `/documents/upload` | Ingest a PDF/image into a case |
| `GET` | `/documents/{id}/status` | Ingestion status polling |
| `GET` | `/documents/cases` | List cases with document counts |
| `GET` | `/documents/cases/{id}/documents` | Documents in a case |
| `POST` | `/checklists/generate` | SSE stream: generate a checklist |
| `GET` | `/checklists/{id}` | Retrieve a checklist with all items |
| `PATCH` | `/checklists/{id}/items/{item_id}` | Edit an item (title/status/rationale) |
| `POST` | `/checklists/{id}/items` | Add an item |
| `DELETE` | `/checklists/{id}/items/{item_id}` | Remove an item |
| `POST` | `/checklists/{id}/finalize` | Trigger pattern extraction |
| `GET` | `/checklists/learned-patterns` | List learned patterns |
| `GET` | `/evidence/{citation_id}` | Retrieve a citation + snippet |

---

## Tests

```bash
uv run pytest                          # unit tests (~5 s)
uv run pytest tests/integration -v    # real Postgres (~30 s)
```

Integration tests run against a real Postgres instance with SAVEPOINT-per-test isolation — no mocked sessions — so constraint violations and schema drift surface in CI rather than production.

---

## Evaluation

### Results by area

| Area | Metric | Result |
|---|---|---|
| Document processing | 8 docs ingested (5 CUAD contracts + 2 degraded + 1 handwritten) | Marker OCR at 0.95+ confidence on clean docs; TrOCR fallback fired on handwritten exhibit — mediocre output that still ranks #2 on semantic queries due to embedding robustness |
| Grounded retrieval | 271 searchable windows across 8 docs; hybrid dense+sparse+RRF | Avg 2.2–2.3 citations per `present` item; handwritten exhibit findable via semantic embedding despite OCR noise |
| Draft quality | 12-item `commercial_contract` template | 10–11 of 12 items resolve to `present` per run; SQL invariant `present ∧ evidence=∅` returns 0 rows — grounding is a hard constraint |
| Improvement loop | 4-run loop closure | 1 pattern promoted at corroboration=3; applied autonomously in Run 4 — mean_edit_distance 1.58 → 0.00, touch_free_rate 91.7% → 100% |

### Loop demonstration

`touch_free_rate` = share of items needing zero operator edits. `pattern_application_rate` = share of items whose `learned_from_pattern_ids` is non-empty.

| Run | edits_applied | mean_edit_distance | touch_free_rate | pattern_application_rate | promoted_patterns |
|---|---|---|---|---|---|
| 1 | 1 | 1.58 | 91.7% | 0.0% | 0 |
| 2 | 1 | 1.58 | 91.7% | 0.0% | 0 |
| 3 | 1 | 1.58 | 91.7% | 0.0% | 1 |
| 4 | 0 | 0.0 | 100.0% | 8.3% | 1 |

Run 3's finalize step promotes the first rule (corroboration crosses 3), but Run 3's generation had already completed before promotion — so pattern_application_rate stays 0% for Run 3. Run 4 starts with the promoted pattern active: `critique` applies it at draft time, no operator edit is needed, and mean_edit_distance drops from 1.58 to 0.00. The loop closed automatically. Raw JSON in `eval/results_loop.md`.

---

## What I built and why

- **Grounding is enforced, not hoped for.** The V1–V5 invariants at the LangGraph validation node reject hallucinated page references and coerce ambiguous evidence to `unclear` rather than letting the LLM fill in the gap. A SQL invariant (`COUNT WHERE status='present' AND evidence=∅`) returns zero rows across every generation — not a policy, a hard constraint.

- **Parent expansion solves the chunking-vs-grounding tension.** Retrieval ranks at the 512-token window level for precision; generation receives the full parent section (up to 3,500 tokens) for context fidelity. Citations still anchor to the precise span — you get signal from dense retrieval and context from the surrounding clause without giving up either.

- **The improvement loop is real, not a diff viewer.** Operator edits (PATCH, DELETE, status corrections) land as typed `EditEvent` rows. A single background LLM call per finalized checklist distills corroborating edits into `LearnedPattern` rules across six pattern types: `rename_rule`, `template_addition`, `template_removal`, `status_default`, `style_preference`, `category_remap`. Promotion is gated on ≥3 corroborating edits and confidence ≥0.7. Promoted rules apply automatically via `load_template` and `critique` on the next run.

- **OCR triage handles both clean and messy inputs.** Marker (Surya) processes printed contracts at roughly 95% confidence. Blocks below 0.6 confidence fall back to `microsoft/trocr-large-handwritten`. The handwritten exhibit's phrase "code for the platform" came through as "code for the patforn" — yet the dense embedding retrieved the document at rank #2 on the query `customer lists pricing models`, because semantic similarity survives OCR noise that exact-match would not.

- **Provider abstraction without vendor lock-in.** The same `init_chat_model` interface switches between Groq `llama-3.3-70b-versatile` and Ollama `qwen3:8b` via a single `LLM_PROVIDER` environment variable. Groq completes a 10–12 item checklist in roughly 60–110 seconds on the free tier; Ollama runs locally for users who don't want to supply API keys.

- **Quality signals are embedded, not bolted on.** Tests run against a real Postgres instance via a SAVEPOINT-per-test fixture — no mocked DB calls. Pydantic schemas split into `Draft` (permissive, for LLM output) and strict (for the API boundary), catching schema drift early. A multi-engineer agent review cycle (data, ML, QA) runs at every phase boundary.

---

## Known limitations and what I'd do next

- **TrOCR quality on the handwritten exhibit is mediocre.** The source photo has enough noise that several phrases come through garbled. A cleaner scan would improve recall on handwriting-specific queries — the retrieval logic is solid, the bottleneck is image quality.

- **Pattern extraction requires edit volume to become useful.** The corroboration threshold (≥3 edits) is intentionally conservative to avoid promoting noise into the prompt. With only the demo documents, patterns promote slowly; a real deployment would see enough operator traffic to fill the loop quickly.

- **Operator identity is wired but unauthenticated.** Every `EditEvent` carries an `actor` field populated from the `X-Operator-Id` header — the surface for a real IdP is in place. Binding it to Cognito, Auth0, or any FastAPI `Depends` provider is a configuration change rather than an architectural one.

- **Groq free-tier throughput limits eval iteration.** At ~100K tokens per day, a back-to-back 4-run eval can exhaust the quota mid-run. The eval harness resets learning state per session so iteration is cheap once quota refreshes; production would use a paid tier or self-hosted endpoint.
