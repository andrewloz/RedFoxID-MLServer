#
# Build targets that correspond to the tags documented in README.md:
#   docker build --target cuda          -t redfoxid/inference-server:cuda .
#   docker build --target openvino-cpu  -t redfoxid/inference-server:openvino-cpu .
#   docker build --target openvino-gpu  -t redfoxid/inference-server:openvino-gpu .
#   docker build --target openvino-npu  -t redfoxid/inference-server:openvino-npu .
#

# ---------- common base (build once) ----------
FROM ubuntu:22.04 AS base

SHELL ["/bin/bash", "-c"]

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR="/tmp/.ultralytics"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        python3.10 \
        python3.10-distutils \
        python3.10-venv \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir ultralytics --no-deps

COPY . /app

RUN mkdir -p "${YOLO_CONFIG_DIR}" && chmod -R 777 "${YOLO_CONFIG_DIR}"

ENTRYPOINT ["/app/docker-entrypoint.sh"]


# ---------- cuda (PyTorch CUDA wheels) ----------
# PyTorch CUDA wheels bundle CUDA; host only needs NVIDIA driver + nvidia-container-runtime
FROM base AS cuda

ARG TORCH_CUDA_TAG=cu126

RUN python -m pip install --no-cache-dir \
        torch \
        torchvision \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

# ---------- Tensorrt (NVIDIA TensorRt wheels) ----------
# host only needs NVIDIA driver + nvidia-container-runtime
FROM cuda AS trt

ARG TORCH_CUDA_TAG
ARG TENSORRT_VERSION="10.13.3.9"

# Consider using tensorrt-lean for smaller image size
RUN python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
    "tensorrt-${TORCH_CUDA_TAG::-1}==${TENSORRT_VERSION}"

# ---------- openvino variants ----------
FROM base AS openvino-base

# OpenVINO doesn't need ultralytics backend, remove it to keep image lean
RUN python -m pip install --no-cache-dir "openvino>=2024.0.0" && \
    python -m pip uninstall -y ultralytics || true

# OpenVINO CPU-only image
FROM openvino-base AS openvino-cpu

# CPU variant requires no additional packages beyond openvino

# OpenVINO GPU image (expects host to pass /dev/dri)
FROM openvino-base AS openvino-gpu

# Bundles Intel dGPU user-space components per https://dgpu-docs.intel.com/driver/client/overview.html
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        lsb-release \
        wget; \
    mkdir -p /usr/share/keyrings; \
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
        gpg --dearmor --batch --yes -o /usr/share/keyrings/intel-graphics.gpg; \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" \
        > /etc/apt/sources.list.d/intel-gpu.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        clinfo \
        intel-gsc \
        intel-metrics-discovery \
        intel-opencl-icd \
        libze-intel-gpu1 \
        libze1; \
    apt-get install -y --no-install-recommends \
        intel-media-va-driver-non-free \
        libmfx-gen1 \
        libvpl-tools \
        libvpl2 \
        libva-glx2 \
        va-driver-all \
        vainfo; \
    apt-get install -y --no-install-recommends \
        intel-ocloc \
        libze-dev; \
    rm -rf /var/lib/apt/lists/*

# OpenVINO NPU image (expects host to pass /dev/accel)
FROM openvino-base AS openvino-npu

# Adds Intel Movidius / NPU runtime components required by OpenVINO
ARG INTEL_NPU_DRIVER_BUNDLE="linux-npu-driver-v1.24.0.20251003-18218973328-ubuntu2404"
ARG INTEL_NPU_DRIVER_URL="https://github.com/intel/linux-npu-driver/releases/download/v1.24.0/${INTEL_NPU_DRIVER_BUNDLE}.tar.gz"
ARG INTEL_LEVEL_ZERO_DEB="level-zero_1.24.2+u24.04_amd64.deb"
ARG INTEL_LEVEL_ZERO_URL="https://github.com/oneapi-src/level-zero/releases/download/v1.24.2/${INTEL_LEVEL_ZERO_DEB}"

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libtbb12 \
        libusb-1.0-0 \
        udev \
        wget; \
    tmpdir="$(mktemp -d)"; \
    wget -q -O "${tmpdir}/linux-npu-driver.tar.gz" "${INTEL_NPU_DRIVER_URL}"; \
    mkdir -p "${tmpdir}/bundle"; \
    tar -xf "${tmpdir}/linux-npu-driver.tar.gz" -C "${tmpdir}/bundle" --strip-components=1; \
    dpkg -i ${tmpdir}/bundle/*.deb || apt-get install -y --no-install-recommends --fix-broken; \
    rm -rf "${tmpdir}/linux-npu-driver.tar.gz"; \
    wget -q -O "${tmpdir}/${INTEL_LEVEL_ZERO_DEB}" "${INTEL_LEVEL_ZERO_URL}"; \
    dpkg -i "${tmpdir}/${INTEL_LEVEL_ZERO_DEB}" || apt-get install -y --no-install-recommends --fix-broken; \
    rm -rf "${tmpdir}"; \
    rm -rf /var/lib/apt/lists/*
