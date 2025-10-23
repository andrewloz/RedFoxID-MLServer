FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 

WORKDIR /app

ENV PATH=$PATH:/home/app/.local/bin

RUN apt-get update && apt-get install -y gpg-agent wget && wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
  gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | \
  tee  /etc/apt/sources.list.d/intel.gpu.jammy.list

RUN apt-get update && apt-get install -y --no-install-recommends build-essential ffmpeg libsm6 libxext6 ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# install deps for runtime layer
RUN python -m pip install -U pip -r requirements.txt 

COPY . /app 

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import socket,sys;s=socket.socket();s.settimeout(2);s.connect(('127.0.0.1',50051));s.close()" || exit 1

CMD ["python", "server.py"]
