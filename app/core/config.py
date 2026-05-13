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

    log_level: str = "INFO"
    env: Literal["dev", "prod"] = "dev"


settings = Settings()
