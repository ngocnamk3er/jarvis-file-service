FROM python:3.11-slim

WORKDIR /app

# This host's IPv6 path is broken (see jarvis-backend/Dockerfile) — same fix.
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini .
COPY migrations ./migrations
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
