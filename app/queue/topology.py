"""The ingestion pipeline's queues — two stages now, declared identically
in jarvis-file-service, jarvis-ocr-worker and jarvis-embedding-worker
(queue args must match byte-for-byte on every side or RabbitMQ rejects a
mismatched re-declare with 406 PRECONDITION_FAILED; keep all three in sync
by hand if this changes).

  ocr.ingest / ocr.ingest.retry / ocr.ingest.dlq
      Stage 1 — extraction + OCR + chunking. jarvis-file-service publishes
      (on upload/reindex), jarvis-ocr-worker consumes.
  ocr.embed / ocr.embed.retry / ocr.embed.dlq
      Stage 2 — embedding + Qdrant indexing. jarvis-ocr-worker publishes
      (after it chunks successfully — the message carries the chunks
      themselves, not just a file_id: unlike stage 1's handoff, there's no
      persistent column to re-derive chunk text/page/boxes from, so the
      message is the only copy until stage 2 embeds it), jarvis-
      embedding-worker consumes.

Both stages use the default (nameless) exchange, routing key = queue name,
and the same retry shape: a message sitting on the `.retry` queue for
RETRY_TTL_MS is dead-lettered back onto its own main queue by RabbitMQ
itself (the standard plugin-free delayed-retry trick) — nack(requeue=True)
was not used for this because it reinserts the *same* message at the
queue head with zero delay, a hot loop against a down dependency rather
than real backoff. After MAX_ATTEMPTS, a plain nack routes straight to
that stage's own dead-letter queue via its main queue's static DLX.

attempt_count lives in Postgres (file_nodes.attempt_count) across both
stages — one shared retry budget for a file's whole trip through the
pipeline, not a separate budget per stage.
"""

INGEST_QUEUE = "ocr.ingest"
INGEST_RETRY_QUEUE = "ocr.ingest.retry"
INGEST_DLQ = "ocr.ingest.dlq"

EMBED_QUEUE = "ocr.embed"
EMBED_RETRY_QUEUE = "ocr.embed.retry"
EMBED_DLQ = "ocr.embed.dlq"

MAX_ATTEMPTS = 3
RETRY_TTL_MS = 10_000


def _declare_stage(channel, queue: str, retry_queue: str, dlq: str) -> None:
    channel.queue_declare(
        queue=queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": dlq,
        },
    )
    channel.queue_declare(
        queue=retry_queue,
        durable=True,
        arguments={
            "x-message-ttl": RETRY_TTL_MS,
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": queue,
        },
    )
    channel.queue_declare(queue=dlq, durable=True)


def declare_topology(channel) -> None:
    """Idempotent — safe to call on every connect, from any of the three
    services, regardless of which stage(s) that service actually uses."""
    _declare_stage(channel, INGEST_QUEUE, INGEST_RETRY_QUEUE, INGEST_DLQ)
    _declare_stage(channel, EMBED_QUEUE, EMBED_RETRY_QUEUE, EMBED_DLQ)
