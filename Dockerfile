# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# This host's IPv6 path is broken (see jarvis-backend/Dockerfile) — same fix.
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

COPY requirements.txt .
# A cache mount rather than --no-cache-dir: this layer is keyed on
# requirements.txt, so adding one package used to re-download all of them,
# and PaddleOCR alone is ~1.5GB over a Docker bridge network measured at
# 2.7 MB/min on this host. The cache lives outside the image, so nothing
# here grows it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY alembic.ini .
COPY migrations ./migrations
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
