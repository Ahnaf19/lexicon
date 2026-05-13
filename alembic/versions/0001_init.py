"""Initial schema: pgvector extension + all PRD §7 tables.

Revision ID: 0001
Revises:
Create Date: 2026-05-13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("case_id", postgresql.UUID(as_uuid=True)),
        sa.Column("original_filename", sa.Text, nullable=False),
        sa.Column("mime", sa.Text),
        sa.Column("sha256", sa.Text, nullable=False),
        sa.Column("doc_type", sa.Text),
        sa.Column("uploaded_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("ocr_engine", sa.Text),
        sa.Column("total_pages", sa.Integer),
        sa.Column("meta", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.UniqueConstraint("sha256", name="uq_documents_sha256"),
    )

    op.create_table(
        "pages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("doc_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("width_px", sa.Integer),
        sa.Column("height_px", sa.Integer),
        sa.Column("ocr_confidence_mean", sa.Float),
        sa.Column("has_handwriting", sa.Boolean, nullable=False, server_default="false"),
    )

    op.execute("""
        CREATE TABLE chunks (
            id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id              uuid NOT NULL REFERENCES documents(id),
            page_number         int  NOT NULL,
            section_heading     text,
            char_offset_start   int  NOT NULL,
            char_offset_end     int  NOT NULL,
            text                text NOT NULL,
            embedding           vector(768) NOT NULL,
            tsv                 tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
            parent_section_id   uuid REFERENCES chunks(id),
            meta                jsonb NOT NULL DEFAULT '{}'::jsonb
        );
    """)
    op.execute("CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);")
    op.execute("CREATE INDEX ON chunks USING GIN (tsv);")
    op.execute("CREATE INDEX ON chunks (doc_id, page_number);")

    op.create_table(
        "checklist_templates",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("doc_type", sa.Text, nullable=False),
        sa.Column("version", sa.Text, nullable=False),
        sa.Column("items", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("active", sa.Boolean, nullable=False, server_default="true"),
    )

    op.create_table(
        "checklists",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("case_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("template_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("checklist_templates.id"), nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="draft"),
        sa.Column("generated_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("finalized_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("model_version", sa.Text),
        sa.Column("prompt_version", sa.Text),
        sa.Column("eval_metrics", postgresql.JSONB),
    )

    op.create_table(
        "checklist_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("checklist_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("checklists.id"), nullable=False),
        sa.Column("source_template_item_id", postgresql.UUID(as_uuid=True)),
        sa.Column("category", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("required", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("confidence", sa.Float),
        sa.Column("rationale", sa.Text),
        sa.Column("learned_from_pattern_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default="{}"),
    )
    op.create_index("ix_checklist_items_checklist_id", "checklist_items", ["checklist_id"])

    op.create_table(
        "evidence_citations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("checklist_item_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("checklist_items.id"), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chunks.id"), nullable=False),
        sa.Column("doc_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("char_offset_start", sa.Integer, nullable=False),
        sa.Column("char_offset_end", sa.Integer, nullable=False),
        sa.Column("snippet", sa.Text, nullable=False),
        sa.Column("retrieval_score", sa.Float, nullable=False),
        sa.Column("rerank_score", sa.Float),
    )
    op.create_index("ix_evidence_citations_checklist_item_id", "evidence_citations", ["checklist_item_id"])

    op.create_table(
        "edit_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("checklist_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("checklists.id"), nullable=False),
        sa.Column("checklist_item_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("checklist_items.id")),
        sa.Column("event_type", sa.Text, nullable=False),
        sa.Column("payload", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("actor", sa.Text),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_edit_events_checklist_id_created_at", "edit_events", ["checklist_id", "created_at"])

    op.execute("""
        CREATE TABLE few_shot_examples (
            id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_type            text NOT NULL,
            category            text NOT NULL,
            template_item_id    uuid,
            original_draft      jsonb NOT NULL,
            final_item          jsonb NOT NULL,
            context_embedding   vector(768) NOT NULL,
            created_at          timestamptz NOT NULL DEFAULT now()
        );
    """)
    op.execute("CREATE INDEX ON few_shot_examples USING hnsw (context_embedding vector_cosine_ops);")

    op.create_table(
        "learned_patterns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("pattern_type", sa.Text, nullable=False),
        sa.Column("doc_type_scope", sa.Text, nullable=False),
        sa.Column("rule_json", postgresql.JSONB, nullable=False),
        sa.Column("supporting_edit_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("corroborating_edit_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("promoted", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("learned_patterns")
    op.execute("DROP TABLE IF EXISTS few_shot_examples;")
    op.drop_index("ix_edit_events_checklist_id_created_at", "edit_events")
    op.drop_table("edit_events")
    op.drop_index("ix_evidence_citations_checklist_item_id", "evidence_citations")
    op.drop_table("evidence_citations")
    op.drop_index("ix_checklist_items_checklist_id", "checklist_items")
    op.drop_table("checklist_items")
    op.drop_table("checklists")
    op.drop_table("checklist_templates")
    op.execute("DROP TABLE IF EXISTS chunks;")
    op.drop_table("pages")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector;")
    op.execute("DROP EXTENSION IF EXISTS pgcrypto;")
