"""Publishes an ingestion task for one newly-uploaded (or reindexed) file.

A new connection per call, not a shared one — called via asyncio.to_thread()
from an async endpoint, which runs this in a thread-pool thread, and pika's
BlockingConnection isn't safe to share across threads. Connecting per
publish sidesteps that entirely for what's an infrequent, cheap operation
(one publish per upload), same tradeoff minio_client makes for its own
blocking SDK calls.
"""

import json

import pika

from app.core.config import settings
from app.queue import topology


def publish_ingest_task(file_id: str) -> None:
    credentials = pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASSWORD)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=settings.RABBITMQ_HOST, port=settings.RABBITMQ_PORT, credentials=credentials,
        )
    )
    try:
        channel = connection.channel()
        topology.declare_topology(channel)
        channel.basic_publish(
            exchange="",
            routing_key=topology.MAIN_QUEUE,
            body=json.dumps({"file_id": file_id}),
            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
        )
    finally:
        connection.close()
