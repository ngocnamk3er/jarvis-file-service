"""create file_nodes

Revision ID: 0001
Revises:
Create Date: 2026-09-05

The per-user folder/file tree (phase 1 of the file workspace feature — see
~/.claude/plans/cosmic-drifting-pinwheel.md). extracted_text/indexing_status
are forward-declared here for phase 2/3 (text extraction + embeddings) but
unused by phase 1's tree-CRUD-only endpoints.
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "file_nodes",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("parent_id", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("minio_object_key", sa.Text(), nullable=True),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("indexing_status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["parent_id"], ["file_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "parent_id", "name", name="uq_file_nodes_user_parent_name"),
        sa.CheckConstraint("type IN ('folder', 'file')", name="ck_file_nodes_type"),
    )
    op.create_index(op.f("ix_file_nodes_user_id"), "file_nodes", ["user_id"], unique=False)
    op.create_index(op.f("ix_file_nodes_parent_id"), "file_nodes", ["parent_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_file_nodes_parent_id"), table_name="file_nodes")
    op.drop_index(op.f("ix_file_nodes_user_id"), table_name="file_nodes")
    op.drop_table("file_nodes")
