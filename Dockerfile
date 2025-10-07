# syntax=docker/dockerfile:1.6
FROM mambaorg/micromamba:1.5.9 AS base
# optional: CUDA base if you need GPUs:
# FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# If you need CUDA + micromamba, you can install micromamba via curl in a CUDA image
# but using the micromamba base is simplest for CPU images.

WORKDIR /app
COPY environment.yml .
# Create env at build time. Name must match your YAML.
RUN micromamba install -y -n cv-inference -f environment.yml && \
    micromamba clean --all --yes

# Make the env the default PATH for all future RUN/CMD
ENV MAMBA_DOCKERFILE_ACTIVATE=1
# (This env var tells the base image’s /usr/local/bin/_entrypoint.sh to auto-activate the env.)
SHELL ["/usr/local/bin/_entrypoint.sh", "bash", "-lc"]

# Install your app
COPY pyproject.toml .
COPY src ./src
RUN python -m pip install -e .

# Non-root user (good hygiene)
USER $MAMBA_USER

# Expose and run
EXPOSE 50051 9090
CMD ["python", "-m", "inference_server.server"]
