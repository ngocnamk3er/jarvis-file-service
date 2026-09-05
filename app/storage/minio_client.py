"""Thin sync wrapper around the `minio` SDK (it has no async client) — see
app/db/repository.py for the asyncio.to_thread() calls that keep these off
the event loop."""

from io import BytesIO
from minio import Minio

from app.core.config import settings

_client: Minio | None = None


def _get_client() -> Minio:
    global _client
    if _client is None:
        client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        if not client.bucket_exists(settings.MINIO_BUCKET):
            client.make_bucket(settings.MINIO_BUCKET)
        _client = client
    return _client


def put_object(object_key: str, content: bytes, content_type: str) -> None:
    _get_client().put_object(
        settings.MINIO_BUCKET,
        object_key,
        BytesIO(content),
        length=len(content),
        content_type=content_type or "application/octet-stream",
    )


def get_object(object_key: str) -> bytes:
    response = _get_client().get_object(settings.MINIO_BUCKET, object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def delete_object(object_key: str) -> None:
    _get_client().remove_object(settings.MINIO_BUCKET, object_key)
