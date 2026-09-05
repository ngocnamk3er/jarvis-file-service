from datetime import datetime
from sqlalchemy import Text, DateTime, ForeignKey, BigInteger, func, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FileNode(Base):
    """One row per folder or file, forming a per-user tree via self-referencing
    parent_id (NULL = root — no row exists for "/" itself).

    Metadata DB owns the tree; MinIO owns bytes (see minio_object_key) —
    Dropbox/Drive-style split, not raw S3 key-prefix folders. A rename/move
    is a pure Postgres update of parent_id/name; MinIO is never touched,
    since objects are keyed by opaque id, never by path."""

    __tablename__ = "file_nodes"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    parent_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("file_nodes.id", ondelete="CASCADE"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)  # "folder" | "file"
    mime_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # f"{user_id}/{id}" — set only for type="file". Never derived from name/
    # path, so renames/moves never touch MinIO.
    minio_object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Plaintext extracted at upload time (phase 2) — powers grep_files
    # directly and is the chunking source for embeddings (phase 3). NULL
    # until ingestion runs, or forever for unsupported/binary types.
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # pending|extracting|embedding|done|failed|skipped — see repository.py.
    # Everything is "pending" until phase 2's ingestion pipeline lands.
    indexing_status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # Backstop only — standard SQL unique constraints treat NULL as
        # distinct from NULL, so this does NOT actually stop two root-level
        # (parent_id IS NULL) items from sharing a name. repository.py does
        # the real check-then-insert at the application layer for that case;
        # this constraint still catches the common (non-root) case even if
        # a caller ever bypasses the repository.
        UniqueConstraint("user_id", "parent_id", "name", name="uq_file_nodes_user_parent_name"),
        CheckConstraint("type IN ('folder', 'file')", name="ck_file_nodes_type"),
    )
