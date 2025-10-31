FROM python:3.9-slim

RUN apt update && apt install -y libgl1 libglib2.0-0

RUN mkdir /tmp/app

RUN pip install ultralytics --no-deps

WORKDIR /app

COPY . /app

ENV YOLO_CONFIG_DIR="/tmp/.ultralytics"
RUN mkdir -p ${YOLO_CONFIG_DIR} && chmod -R 777 ${YOLO_CONFIG_DIR}

RUN pip install -r requirements-base.txt

# openvino specific deps
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip3 install openvino>=2024.0.0

CMD ["python", "server.py", "config.ini"]
