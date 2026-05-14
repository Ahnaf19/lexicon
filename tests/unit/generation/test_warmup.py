"""Unit tests for warmup_llm() provider-branching logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.generation.warmup import warmup_llm


@pytest.fixture(autouse=True)
def reset_warmed():
    """Reset module-level _warmed flag between tests."""
    import app.generation.warmup as warmup_mod
    warmup_mod._warmed = False
    yield
    warmup_mod._warmed = False


@pytest.mark.asyncio
async def test_groq_warmup_makes_no_network_calls():
    """With LLM_PROVIDER=groq, warmup_llm() must not invoke the chat model."""
    mock_get = AsyncMock()
    with (
        patch("app.generation.warmup.settings") as mock_settings,
        patch("app.generation.warmup.get_chat_model", mock_get),
    ):
        mock_settings.llm_provider = "groq"
        await warmup_llm()

    mock_get.assert_not_called()


@pytest.mark.asyncio
async def test_groq_warmup_sets_warmed_flag():
    import app.generation.warmup as warmup_mod

    with (
        patch("app.generation.warmup.settings") as mock_settings,
        patch("app.generation.warmup.get_chat_model", AsyncMock()),
    ):
        mock_settings.llm_provider = "groq"
        await warmup_llm()

    assert warmup_mod._warmed is True


@pytest.mark.asyncio
async def test_ollama_warmup_invokes_model():
    """With LLM_PROVIDER=ollama, warmup_llm() calls the chat model once."""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value="ok")
    mock_get = MagicMock(return_value=mock_model)  # get_chat_model is sync

    with (
        patch("app.generation.warmup.settings") as mock_settings,
        patch("app.generation.warmup.get_chat_model", mock_get),
    ):
        mock_settings.llm_provider = "ollama"
        await warmup_llm()

    mock_get.assert_called_once_with(role="quality")
    mock_model.ainvoke.assert_called_once_with("ok")


@pytest.mark.asyncio
async def test_warmup_idempotent():
    """Calling warmup_llm() twice only invokes the model once."""
    import app.generation.warmup as warmup_mod

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value="ok")
    mock_get = MagicMock(return_value=mock_model)  # get_chat_model is sync

    with (
        patch("app.generation.warmup.settings") as mock_settings,
        patch("app.generation.warmup.get_chat_model", mock_get),
    ):
        mock_settings.llm_provider = "ollama"
        await warmup_llm()
        await warmup_llm()

    mock_get.assert_called_once()
