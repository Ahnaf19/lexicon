"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    yield


app = FastAPI(title="Lexicon", version="0.1.0", lifespan=lifespan)
app.include_router(documents_router, prefix="/documents", tags=["documents"])
