"""The ingestion queue's shape — declared here and in jarvis-ocr-worker's
own copy of this exact file (queue args must match byte-for-byte on both
sides or RabbitMQ rejects a mismatched re-declare with 406
PRECONDITION_FAILED; keep the two in sync by hand if either changes).

Three queues, default (nameless) exchange, routing key = queue name — a
named exchange would add a hop with no benefit for a strict 1:1 task queue:

  ocr.ingest        the real work queue. A message that exhausts retries
                     (see MAX_ATTEMPTS) is nacked without requeue, and this
                     queue's own static DLX routes it straight to
                     ocr.ingest.dlq automatically — no manual publish
                     needed for that path.
  ocr.ingest.retry   pure delay buffer. No consumer ever attaches to it —
                     a message sitting here for RETRY_TTL_MS is dead-
                     lettered back onto ocr.ingest by RabbitMQ itself, the
                     standard plugin-free delayed-retry trick. The worker
                     explicitly publishes here (then acks the original off
                     ocr.ingest) for a retryable failure under
                     MAX_ATTEMPTS — nack(requeue=True) was not used for
                     this because it reinserts the *same* message at the
                     queue head with zero delay, a hot loop against a down
                     dependency rather than real backoff.
  ocr.ingest.dlq     terminal parking queue — no TTL, no consumer. An
                     operator inspects/replays it manually.

attempt_count itself lives in Postgres (file_nodes.attempt_count), not in
an AMQP header — one `UPDATE ... RETURNING attempt_count` is the single
source of truth, trivially inspectable with psql, and survives a broker
restart even though the message itself is also durable.
"""

MAIN_QUEUE = "ocr.ingest"
RETRY_QUEUE = "ocr.ingest.retry"
DLQ = "ocr.ingest.dlq"

MAX_ATTEMPTS = 3
RETRY_TTL_MS = 10_000


def declare_topology(channel) -> None:
    """Idempotent — safe to call on every connect, from either side."""
    channel.queue_declare(
        queue=MAIN_QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": DLQ,
        },
    )
    channel.queue_declare(
        queue=RETRY_QUEUE,
        durable=True,
        arguments={
            "x-message-ttl": RETRY_TTL_MS,
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": MAIN_QUEUE,
        },
    )
    channel.queue_declare(queue=DLQ, durable=True)
