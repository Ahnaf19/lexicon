# Setup & Running

Complete installation and run guide for all platforms and configurations.

---

## Prerequisites

| Tool | Install |
|---|---|
| **Docker Desktop** | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) — includes Docker Compose |
| **uv** (Python 3.12 package manager) | macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| | Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| **Groq API key** (free) | [console.groq.com](https://console.groq.com) — create an account, copy the key |

---

## 1. Clone and configure

```bash
git clone https://github.com/Ahnaf19/lexicon.git
cd lexicon

cp .env.template .env
```

Open `.env` and fill in the required value:

```bash
# Required
GROQ_API_KEY=gsk_...your_key_here...

# Everything else has a working default — only change if you need to
```

### Full `.env` reference

```bash
# Postgres — matches the docker-compose.yml credentials
DB_URL=postgresql+asyncpg://lexicon:lexicon@localhost:5433/lexicon

# LLM provider: "groq" (default, fast, free-tier) or "ollama" (local, no API key)
LLM_PROVIDER=groq
GROQ_API_KEY=
GROQ_MODEL_QUALITY=llama-3.3-70b-versatile
GROQ_MODEL_FAST=llama-3.1-8b-instant
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_QUALITY=qwen3:8b
OLLAMA_MODEL_FAST=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text

# Langfuse tracing (optional — leave blank to disable)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000

# Runtime
LOG_LEVEL=INFO
ENV=dev
```

> [!NOTE]
> Postgres runs on host port **5433** (not 5432) because 5432 may be occupied by a local Postgres install. The `DB_URL` default already reflects this.

---

## 2. Install Python dependencies

```bash
uv sync
```

This creates a `.venv` at the project root and installs all locked dependencies. No `pip install` needed.

---

## 3. Run paths

### Option A — Postgres only (recommended for local development)

Start only Postgres. Run FastAPI and Streamlit directly on the host.

```bash
# Start Postgres
docker compose up -d postgres

# Run migrations (once per clone, or after any schema change)
uv run alembic upgrade head

# Ingest sample documents
uv run python -m app.cli ingest samples/clean samples/degraded samples/handwritten

# Start the API (in one terminal)
uv run uvicorn app.main:app --reload
# or: make api

# Start the UI (in another terminal)
uv run streamlit run ui/streamlit_app.py
# or: make ui
```

Ports: API on `:8000`, UI on `:8501`.

> [!TIP]
> On macOS, prefix long-running processes with `caffeinate -i` to prevent App Nap from throttling Ollama or the LLM mid-run.

### Option B — Full Docker stack

Bring up every service including the API and UI as containers.

```bash
docker compose up -d

# Run migrations against the containerised DB
docker compose exec api uv run alembic upgrade head

# Ingest samples from inside the container
docker compose exec api uv run python -m app.cli ingest samples/clean samples/degraded samples/handwritten
```

API: `http://localhost:8000` · UI: `http://localhost:8501`

### Option C — With local LLM (Ollama)

Switch to local embeddings and local LLM inference. Requires pulling the models first.

```bash
# Start Postgres + Ollama
docker compose --profile local-llm up -d

# Pull the required models
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen3:8b  # optional — only needed for LLM_PROVIDER=ollama

# Update .env
LLM_PROVIDER=ollama          # for LLM generation
EMBEDDING_MODEL=nomic-embed-text  # already the default
```

> [!NOTE]
> Groq (`LLM_PROVIDER=groq`) uses Ollama only for embeddings — you still need `nomic-embed-text` pulled regardless of LLM provider.

### Option D — With Langfuse observability

```bash
docker compose --profile obs up -d
```

Langfuse UI opens at `http://localhost:3000`. Create a project, copy the public + secret keys into `.env`, then restart the API. All LangGraph node calls will appear as traces.

---

## 4. CLI reference

```bash
# Ingest one or more directories
uv run python -m app.cli ingest <dir1> [dir2 ...]

# Generate a checklist via CLI (bypasses the UI)
uv run python -m app.cli checklist generate \
  --case-id 00000000-0000-0000-0000-000000000001 \
  --template commercial_contract

# Run the learning-loop evaluation (4 runs, resets state)
uv run python -m eval.run
```

---

## 5. Tests

```bash
# Unit tests (~5 s, no external deps)
uv run pytest

# Integration tests (requires running Postgres on 5433)
uv run pytest tests/integration -v
```

Integration tests create a `lexicon_test` database and use SAVEPOINT-per-test isolation — no mocked sessions.

---

## 6. Make targets

```bash
make install    # uv sync
make migrate    # alembic upgrade head
make test       # uv run pytest
make api        # uvicorn app.main:app --reload
make ui         # streamlit run ui/streamlit_app.py
make eval       # python -m eval.run
make lint       # ruff check app tests
make format     # ruff format app tests
```

---

## 7. Windows (WSL2 + Docker Desktop)

Docker Desktop on Windows uses WSL2 as its backend. All commands in this guide run identically in a WSL2 terminal once Docker Desktop is installed.

1. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) — enable the WSL2 backend during setup.
2. Open a WSL2 terminal (Windows Terminal → Ubuntu, or the Docker Desktop terminal).
3. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. Follow the steps above from "Clone and configure" onward — they work verbatim.

---

## 8. NVIDIA GPU (RTX 3060 Ti and other CUDA cards)

Ollama uses your GPU for embedding inference and local LLM generation when the NVIDIA Container Toolkit is configured.

### Install the container toolkit (WSL2 Ubuntu)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify

```bash
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
# Should print your GPU name and driver version
```

### Start Ollama with GPU access

```bash
docker compose --profile local-llm up -d ollama
```

The `deploy.resources.reservations.devices` block in `docker-compose.yml` is already wired. No further config needed.

> [!TIP]
> Confirm Ollama is using the GPU by watching its logs during model load:
> ```bash
> docker compose logs -f ollama
> # Look for: gpu=0 name=NVIDIA GeForce RTX 3060 Ti vram=7.67GiB
> #           layers_on_gpu=33
> ```
> You can also run `nvidia-smi` while a generation is in progress — look for the `ollama` process in the GPU Process column.

### Performance: RTX 3060 Ti vs Apple M3 Pro

| Task | M3 Pro (18 GB unified) | RTX 3060 Ti (8 GB VRAM) |
|---|---|---|
| `nomic-embed-text` | ~180 tokens/s | ~400–600 tokens/s |
| `qwen3:8b` generation | ~30–40 tokens/s | ~55–80 tokens/s |
| Marker OCR (10-page PDF) | ~8–12 s | ~2–4 s |

The RTX 3060 Ti's dedicated VRAM gives roughly 2× faster embedding throughput and 1.5–2× faster local LLM generation. The M3 Pro's advantage is zero model-load latency (no PCIe transfer) and higher sustained CPU bandwidth for pre/post-processing.

> [!NOTE]
> Groq (`LLM_PROVIDER=groq`) runs at ~400 tokens/s regardless of local GPU — it's faster than both local options for generation. GPU acceleration primarily benefits embeddings (`nomic-embed-text`) and the local LLM fallback (`qwen3:8b`).

---

## 9. Troubleshooting

**Port 5432 already in use**

The compose file maps Postgres to host port 5433 specifically to avoid this. If 5433 is also taken, change the `ports:` mapping in `docker-compose.yml` and update `DB_URL` in `.env` to match.

**`GROQ_API_KEY` not set**

```
KeyError: 'GROQ_API_KEY'
```

Add `GROQ_API_KEY=gsk_...` to `.env`. If you don't have a key, set `LLM_PROVIDER=ollama` and ensure Ollama is running with `qwen3:8b` pulled.

**Groq free-tier quota exhausted**

The free tier provides ~100K tokens/day. A full 4-run eval consumes ~80K tokens. If you hit the limit mid-run, the eval script will error with a rate-limit response from Groq. Wait for the quota to reset (daily) or switch to `LLM_PROVIDER=ollama`.

**Ollama model not found**

```
Error: model 'nomic-embed-text' not found
```

Run: `docker compose exec ollama ollama pull nomic-embed-text`

**OCR running out of memory**

Marker and TrOCR each load ~2–4 GB of model weights. On machines with < 16 GB RAM, running both simultaneously during ingestion of large documents can trigger OOM. If you see `MemoryError` during ingestion:
- Ingest one document at a time rather than a full directory.
- Increase Docker Desktop's memory limit (Settings → Resources → Memory).

**Alembic `head` is already up to date**

Not an error. Alembic is idempotent — running `upgrade head` on an already-migrated DB is safe.
