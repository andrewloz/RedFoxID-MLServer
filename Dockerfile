
# ---------- Stage 1: Builder ----------
FROM python:3.9-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

# Install all deps (Torch + OpenVINO + Ultralytics)
RUN pip install --no-cache-dir ultralytics --no-deps && \
    pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir openvino>=2024.0.0 && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR="/tmp/.ultralytics"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy only installed site-packages and Python binaries from builder
COPY --from=builder /usr/local /usr/local

WORKDIR /app
COPY . /app

RUN mkdir -p ${YOLO_CONFIG_DIR} && chmod -R 777 ${YOLO_CONFIG_DIR}

CMD ["python", "server.py", "config.ini"]

