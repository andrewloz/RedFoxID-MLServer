FROM python:3.9-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# install deps for runtime layer
RUN python -m pip install -U pip wheel setuptools \
 && python -m pip wheel --wheel-dir=/wheels -r requirements.txt

COPY . /app 

FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ffmpeg and X libs often used by Ultralytics for image I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged user WITH a home directory
RUN useradd -m -d /home/app -s /usr/sbin/nologin app && mkdir -p /home/app/.config/Ultralytics
ENV HOME=/home/app YOLO_CONFIG_DIR=/home/app/.config/Ultralytics

COPY --from=builder /wheels /wheels
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

COPY --from=builder /app /app

ENV PATH=$PATH:/home/app/.local/bin

USER app

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import socket,sys;s=socket.socket();s.settimeout(2);s.connect(('127.0.0.1',50051));s.close()" || exit 1

CMD ["python", "server.py"]
