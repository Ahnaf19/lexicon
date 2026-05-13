"""Tenacity retry decorators for network I/O and LLM calls per PRD §6.3."""

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)


def _is_retryable_llm_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503)
    return isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError))


network_retry: Callable[..., Any] = retry(
    retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=8) + wait_random(0, 1),
    reraise=True,
)

llm_retry: Callable[..., Any] = retry(
    retry=retry_if_exception(_is_retryable_llm_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=8) + wait_random(0, 1),
    reraise=True,
)
