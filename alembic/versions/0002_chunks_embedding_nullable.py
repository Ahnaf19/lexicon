"""Make chunks.embedding nullable so phase 2 ingestion can write blocks without embeddings.

Phase 3 (retrieval) will re-chunk and populate embeddings in one pass.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-13
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding DROP NOT NULL;")


def downgrade() -> None:
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding SET NOT NULL;")
