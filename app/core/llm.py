"""LLM provider abstraction — returns a LangChain BaseChatModel per PRD §5g."""

from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.core.config import settings


def get_chat_model(role: Literal["fast", "quality"] = "quality") -> BaseChatModel:
    kwargs: dict[str, Any] = {"temperature": 0, "max_retries": 3, "timeout": 60}
    if settings.llm_provider == "groq":
        model = settings.groq_model_quality if role == "quality" else settings.groq_model_fast
        model_name = f"groq:{model}"
        if settings.groq_api_key is not None:
            kwargs["groq_api_key"] = settings.groq_api_key.get_secret_value()
    else:
        model = settings.ollama_model_quality if role == "quality" else settings.ollama_model_fast
        model_name = f"ollama:{model}"
        kwargs["base_url"] = str(settings.ollama_base_url)
    return init_chat_model(model_name, **kwargs)
