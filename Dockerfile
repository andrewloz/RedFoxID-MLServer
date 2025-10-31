#
# Build targets that correspond to the tags documented in README.md:
#   docker build --target production    -t redfoxid/inference-server:production .
#   docker build --target cuda          -t redfoxid/inference-server:cuda .
#   docker build --target openvino-cpu  -t redfoxid/inference-server:openvino-cpu .
#   docker build --target openvino-gpu  -t redfoxid/inference-server:openvino-gpu .
#   docker build --target openvino-npu  -t redfoxid/inference-server:openvino-npu .
#

# ---------- common base (build once) ----------
FROM python:3.9-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR="/tmp/.ultralytics"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ultralytics --no-deps

COPY . /app

RUN mkdir -p "${YOLO_CONFIG_DIR}" && chmod -R 777 "${YOLO_CONFIG_DIR}"

CMD ["python", "server.py", "config.ini"]


# ---------- production (portable CPU / ONNX Runtime) ----------
FROM base AS production

RUN pip install --no-cache-dir "onnxruntime>=1.18" && \
    pip uninstall -y torch torchvision || true


# ---------- cuda (PyTorch CUDA wheels) ----------
# PyTorch CUDA wheels bundle CUDA; host only needs NVIDIA driver + nvidia-container-runtime
FROM base AS cuda

ARG TORCH_CUDA_TAG=cu121

RUN pip install --no-cache-dir \
        "torch==2.*+${TORCH_CUDA_TAG}" \
        "torchvision==0.*+${TORCH_CUDA_TAG}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"


# ---------- openvino variants ----------
FROM base AS openvino-base

RUN pip install --no-cache-dir "openvino>=2024.0.0" && \
    pip uninstall -y torch torchvision onnxruntime || true

# OpenVINO CPU-only image
FROM openvino-base AS openvino-cpu

# CPU variant requires no additional packages beyond openvino

# OpenVINO GPU image (expects host to pass /dev/dri)
FROM openvino-base AS openvino-gpu

# GPU-specific drivers remain host provided; container inherits OpenVINO runtime

# OpenVINO NPU image (expects host to pass /dev/accel)
FROM openvino-base AS openvino-npu

# NPU variant reuses the same OpenVINO runtime setup
