"""add ingest tracking columns

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-11

Splits the old single `embedding` status into its real sub-steps and adds
the columns an operator needs to actually debug a failed/stuck file —
which stage it was on, why it failed, how many times it's been retried,
and when the current stage started (to spot a row stuck mid-stage). Purely
additive: the old 6 status values are a subset of the new 8, so existing
rows need no backfill.
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

_STATUSES = (
    "pending", "extracting", "chunking", "embedding", "indexing",
    "done", "failed", "skipped",
)


def upgrade() -> None:
    op.add_column("file_nodes", sa.Column("error_stage", sa.Text(), nullable=True))
    op.add_column("file_nodes", sa.Column("error_message", sa.Text(), nullable=True))
    op.add_column(
        "file_nodes",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "file_nodes", sa.Column("stage_started_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_check_constraint(
        "ck_file_nodes_indexing_status",
        "file_nodes",
        "indexing_status IN (" + ", ".join(f"'{s}'" for s in _STATUSES) + ")",
    )


def downgrade() -> None:
    op.drop_constraint("ck_file_nodes_indexing_status", "file_nodes", type_="check")
    op.drop_column("file_nodes", "stage_started_at")
    op.drop_column("file_nodes", "attempt_count")
    op.drop_column("file_nodes", "error_message")
    op.drop_column("file_nodes", "error_stage")
