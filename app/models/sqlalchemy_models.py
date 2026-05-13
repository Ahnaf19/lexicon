"""SQLAlchemy 2 ORM — mirrors the Alembic schema; Base.metadata is the autogen target."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    mime: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type: Mapped[str | None] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    ocr_engine: Mapped[str | None] = mapped_column(Text)
    total_pages: Mapped[int | None] = mapped_column(Integer)
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)  # type: ignore[type-arg]

    __table_args__ = (UniqueConstraint("sha256", name="uq_documents_sha256"),)

    pages: Mapped[list[Page]] = relationship(back_populates="document")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="document")


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    width_px: Mapped[int | None] = mapped_column(Integer)
    height_px: Mapped[int | None] = mapped_column(Integer)
    ocr_confidence_mean: Mapped[float | None] = mapped_column(Float)
    has_handwriting: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    document: Mapped[Document] = relationship(back_populates="pages")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    section_heading: Mapped[str | None] = mapped_column(Text)
    char_offset_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_offset_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)  # type: ignore[type-arg]
    # tsv is a GENERATED column — managed by Alembic DDL, not the ORM
    parent_section_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chunks.id"))
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)  # type: ignore[type-arg]

    document: Mapped[Document] = relationship(back_populates="chunks")


class ChecklistTemplate(Base):
    __tablename__ = "checklist_templates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    items: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)  # type: ignore[type-arg]
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Checklist(Base):
    __tablename__ = "checklists"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    template_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("checklist_templates.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")
    generated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    finalized_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    model_version: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    eval_metrics: Mapped[dict | None] = mapped_column(JSONB)  # type: ignore[type-arg]

    items: Mapped[list[ChecklistItem]] = relationship(back_populates="checklist")


class ChecklistItem(Base):
    __tablename__ = "checklist_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    checklist_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("checklists.id"), nullable=False)
    source_template_item_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    category: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text)
    learned_from_pattern_ids: Mapped[list[uuid.UUID]] = mapped_column(  # type: ignore[type-arg]
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )

    checklist: Mapped[Checklist] = relationship(back_populates="items")
    citations: Mapped[list[EvidenceCitation]] = relationship(back_populates="checklist_item")


class EvidenceCitation(Base):
    __tablename__ = "evidence_citations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    checklist_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("checklist_items.id"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chunks.id"), nullable=False)
    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    char_offset_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_offset_end: Mapped[int] = mapped_column(Integer, nullable=False)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_score: Mapped[float] = mapped_column(Float, nullable=False)
    rerank_score: Mapped[float | None] = mapped_column(Float)

    checklist_item: Mapped[ChecklistItem] = relationship(back_populates="citations")


class EditEvent(Base):
    __tablename__ = "edit_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    checklist_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("checklists.id"), nullable=False)
    checklist_item_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("checklist_items.id"))
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)  # type: ignore[type-arg]
    actor: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)


class FewShotExample(Base):
    __tablename__ = "few_shot_examples"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_type: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    template_item_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    original_draft: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    final_item: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    context_embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)  # type: ignore[type-arg]
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)


class LearnedPattern(Base):
    __tablename__ = "learned_patterns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_type: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type_scope: Mapped[str] = mapped_column(Text, nullable=False)
    rule_json: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    supporting_edit_ids: Mapped[list[uuid.UUID]] = mapped_column(  # type: ignore[type-arg]
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    corroborating_edit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
