# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# This host's IPv6 path is broken (see jarvis-backend/Dockerfile) — same fix.
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

# Three shared libraries -slim omits that PaddleOCR's stack needs, each
# of which fails somewhere that does not mention OCR:
#   libgl1, libglib2.0-0  — opencv-contrib-python links against them, and
#       the import dies at "libGL.so.1: cannot open shared object file".
#   libgomp1              — paddle's own libpaddle.so needs the GNU OpenMP
#       runtime. Its absence surfaces as paddle's "Can not import paddle
#       core while this file exists: .../libpaddle.so", which points at a
#       file that is present and says nothing about the one that is not;
#       `ldd` on that .so is what actually names it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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

# Bake the OCR weights in. PaddleOCR otherwise fetches them on first use —
# ~50s of latency on whichever upload happens to be first after a pod
# starts, and a hard runtime dependency on reaching PaddlePaddle's model
# host, which a NetworkPolicy would be right to forbid. Building them in
# turns both into a build-time cost. Failure is tolerated so a model host
# having a bad day cannot block a deploy; the runtime download then
# remains as the fallback it already is.
RUN python3 -c "\
from app.services.layout import _get_ocr; _get_ocr()" || true

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
