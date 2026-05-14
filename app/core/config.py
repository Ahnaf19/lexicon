"""Settings singleton — reads .env; all app code imports from here, never os.environ."""

from typing import Literal

from pydantic import AnyHttpUrl, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_url: PostgresDsn = PostgresDsn("postgresql+asyncpg://lexicon:lexicon@localhost:5432/lexicon")

    llm_provider: Literal["groq", "ollama"] = "groq"
    groq_api_key: SecretStr | None = None
    groq_model_quality: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    ollama_model_quality: str = "qwen3:8b"
    ollama_model_fast: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: AnyHttpUrl | None = None

    # Per-call-site LLM timeouts (seconds). Separate knobs because foreground
    # (draft/critique) and background (extract) have different SLOs.
    llm_draft_timeout_s: int = 120
    llm_critique_timeout_s: int = 120
    llm_extract_timeout_s: int = 300

    log_level: str = "INFO"
    env: Literal["dev", "prod"] = "dev"

    auto_index: bool = True
    embedding_dim: int = 768

    trocr_printed_threshold: float = 0.4


settings = Settings()

# ---------------------------------------------------------------------------
# Propagate device-optimised Surya batch sizes to os.environ *before* Marker
# is imported for the first time. Surya reads its own Settings() from os.environ
# at import time (looks for local.env, not .env), so we must write here.
# setdefault → operator can still override via shell env.
# ---------------------------------------------------------------------------
import os as _os  # noqa: E402

import torch as _torch  # noqa: E402


def _detect_device() -> str:
    if _torch.cuda.is_available():
        return "cuda"
    if getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DEVICE = _detect_device()

_os.environ.setdefault("TORCH_DEVICE", _DEVICE)
if _DEVICE in ("mps", "cpu"):
    _os.environ.setdefault("DETECTOR_BATCH_SIZE", "8")
    _os.environ.setdefault("RECOGNITION_BATCH_SIZE", "32")
    # LAYOUT_BATCH_SIZE is intentionally left at Surya's default — raising it on MPS
    # triggers an index-out-of-bounds in the vision encoder for large/unusual page sizes.
    _os.environ.setdefault("TABLE_REC_BATCH_SIZE", "8")
    _os.environ.setdefault("OCR_ERROR_BATCH_SIZE", "16")
if _DEVICE == "mps":
    # Default watermark ratio is ~0.5 on Apple Silicon (18 GB on 36 GB machines),
    # which causes OOM when running multiple large docs in sequence. Setting to 0.0
    # disables the cap so PyTorch can use all available unified memory.
    _os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
