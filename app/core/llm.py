"""LLM provider abstraction — returns a LangChain BaseChatModel per PRD §5g."""

from functools import lru_cache
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from loguru import logger

from app.core.config import settings


@lru_cache(maxsize=4)
def get_chat_model(role: Literal["fast", "quality"] = "quality") -> BaseChatModel:
    # Set the httpx-level timeout well above any asyncio.wait_for ceiling so the
    # inner wait_for is always the binding deadline (a plain httpx.ReadTimeout has
    # a different exception shape and is harder to log uniformly).
    timeout = 300 if settings.llm_provider == "ollama" else 60
    kwargs: dict[str, Any] = {"temperature": 0, "max_retries": 3, "timeout": timeout}
    if settings.llm_provider == "groq":
        model = settings.groq_model_quality if role == "quality" else settings.groq_model_fast
        model_name = f"groq:{model}"
        if settings.groq_api_key is not None:
            kwargs["groq_api_key"] = settings.groq_api_key.get_secret_value()
    else:
        model = settings.ollama_model_quality if role == "quality" else settings.ollama_model_fast
        model_name = f"ollama:{model}"
        kwargs["base_url"] = str(settings.ollama_base_url)
        kwargs["num_ctx"] = 8192
        chat = init_chat_model(model_name, **kwargs)
        logger.bind(model=model, num_ctx=kwargs["num_ctx"]).info("ollama_chat_model_initialized")
        return chat
    return init_chat_model(model_name, **kwargs)
