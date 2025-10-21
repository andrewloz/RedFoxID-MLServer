FROM python:3.9-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . ./ 
COPY requirements.txt ./


RUN python -m pip install -U pip wheel setuptools \
 && python -m pip wheel --wheel-dir=/wheels -r requirements.txt


FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app

COPY requirements.txt ./requirements.txt

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY --from=builder /app ./

USER app

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import socket,sys;s=socket.socket();s.settimeout(2);s.connect(('127.0.0.1',50051));s.close()" || exit 1

CMD ["python", "server.py"]