"""Shared pytest fixtures — fake DB for unit tests; real Postgres for integration tests."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import BaseModel

from app.ingestion.models import Block
from app.models.pydantic_models import BBox, DocumentMeta


# ---------------------------------------------------------------------------
# FakeListChatModel fixture — stubs app.core.llm.get_chat_model
# ---------------------------------------------------------------------------


def make_fake_chat_model(responses: list[str]) -> FakeListChatModel:
    """Build a FakeListChatModel that cycles through JSON string responses."""
    return FakeListChatModel(responses=responses)


@pytest.fixture
def patch_llm(monkeypatch: pytest.MonkeyPatch):
    """Replace get_chat_model with a factory returning a FakeListChatModel.

    Tests that need specific responses should use make_fake_chat_model directly
    and monkeypatch app.core.llm.get_chat_model themselves.
    """
    fake = make_fake_chat_model(['{"status":"unclear","confidence":0.5,"rationale":"stub","cited_evidence":[]}'])

    def _factory(role: str = "quality") -> FakeListChatModel:
        return fake

    monkeypatch.setattr("app.core.llm.get_chat_model", _factory)
    return fake


# ---------------------------------------------------------------------------
# Real-Postgres session fixture (integration tests only)
# ---------------------------------------------------------------------------

try:
    import asyncpg  # noqa: F401
    _ASYNCPG_AVAILABLE = True
except ImportError:
    _ASYNCPG_AVAILABLE = False


@pytest.fixture(scope="session")
def pg_engine():
    """Session-scoped async engine pointing at lexicon_test.

    Skips if Postgres is unreachable so unit tests still pass in isolation.
    """
    import asyncio

    import sqlalchemy
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.core.config import settings

    test_url = str(settings.db_url).replace("/lexicon", "/lexicon_test")

    async def _try_connect():
        engine = create_async_engine(test_url, echo=False)
        try:
            async with engine.connect() as conn:
                await conn.execute(sqlalchemy.text("SELECT 1"))
            return engine
        except Exception as exc:
            await engine.dispose()
            raise exc

    try:
        engine = asyncio.run(_try_connect())
    except Exception as exc:
        pytest.skip(f"Postgres unreachable for integration tests: {exc}")
        return  # never reached

    yield engine

    asyncio.run(engine.dispose())


@pytest.fixture(scope="session")
def pg_schema(pg_engine):
    """Run alembic upgrade head once per session on lexicon_test."""
    import asyncio

    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    test_url = str(pg_engine.url).replace("+asyncpg", "")
    cfg.set_main_option("sqlalchemy.url", test_url)
    command.upgrade(cfg, "head")
    return pg_engine


@pytest.fixture
def pg_session(pg_schema):
    """Per-test async session factory with SAVEPOINT rollback for isolation."""
    from sqlalchemy.ext.asyncio import AsyncSession

    async def _get():
        async with AsyncSession(pg_schema) as session:
            async with session.begin():
                yield session
                await session.rollback()

    return _get


# ---------------------------------------------------------------------------
# Fake DB session
# ---------------------------------------------------------------------------


class FakeSession:
    """Records add/commit calls; exposes added rows by model class name."""

    def __init__(self) -> None:
        self.added: dict[str, list[Any]] = defaultdict(list)
        self._committed = False

    def add(self, obj: Any) -> None:
        self.added[type(obj).__name__].append(obj)

    async def commit(self) -> None:
        self._committed = True

    async def get(self, model: Any, pk: Any) -> Any | None:
        rows = self.added.get(model.__name__, [])
        for row in rows:
            if getattr(row, "id", None) == pk:
                return row
        return None

    async def execute(self, stmt: Any) -> Any:
        result = MagicMock()
        result.scalars.return_value.first.return_value = None
        result.scalar_one.return_value = 0
        return result

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@asynccontextmanager
async def fake_session_factory_ctx() -> AsyncGenerator[FakeSession, None]:
    yield FakeSession()


class FakeSessionFactory:
    """Callable that returns a new FakeSession each time (usable as async_sessionmaker)."""

    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def __call__(self) -> FakeSession:
        s = FakeSession()
        self.sessions.append(s)
        return s

    @property
    def last(self) -> FakeSession:
        return self.sessions[-1]


@pytest.fixture
def fake_sf() -> FakeSessionFactory:
    return FakeSessionFactory()


# ---------------------------------------------------------------------------
# Block factory
# ---------------------------------------------------------------------------


def make_block(
    text: str = "Sample legal text.",
    confidence: float = 0.9,
    is_handwriting: bool = False,
    page: int = 1,
) -> Block:
    return Block(
        text=text,
        page=page,
        bbox=BBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0),
        char_offset_start=0,
        char_offset_end=len(text),
        confidence=confidence,
        ocr_engine="marker",
        is_handwriting=is_handwriting,
    )


# ---------------------------------------------------------------------------
# DocumentMeta factory
# ---------------------------------------------------------------------------


def make_meta(doc_id: uuid.UUID | None = None, doc_type: str = "license") -> DocumentMeta:
    return DocumentMeta(
        doc_id=doc_id or uuid.uuid4(),
        doc_type=doc_type,  # type: ignore[arg-type]
        parties=[],
        effective_date=None,
        monetary_terms=[],
        defined_terms=[],
        exhibits_referenced=[],
        signature_blocks=[],
        governing_law=None,
        confidence=0.85,
    )


# ---------------------------------------------------------------------------
# PIL stub image
# ---------------------------------------------------------------------------


def make_pil_stub() -> Any:
    """Minimal PIL Image stub (does not require pillow display)."""
    img = MagicMock()
    img.convert.return_value = img
    return img


# ---------------------------------------------------------------------------
# MarkerOutput stub
# ---------------------------------------------------------------------------

from app.ingestion.marker_runner import MarkerOutput  # noqa: E402


def make_marker_output(blocks: list[Block]) -> MarkerOutput:
    marker_blocks = []
    for _ in blocks:
        mb = MagicMock()
        mb.get_image.return_value = make_pil_stub()
        marker_blocks.append(mb)
    document_stub = MagicMock()
    return MarkerOutput(
        blocks=blocks,
        _marker_blocks=marker_blocks,
        _document=document_stub,
    )
