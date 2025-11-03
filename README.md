RedfoxID Inference Server
=========================

This repository contains the gRPC inference server that powers RedfoxID computer-vision workloads. The server loads one or more ONNX object-detection models and exposes a simple API for making predictions from PNG images.

Quick Start
-----------

1. Ensure Python `3.9` is installed and available on your `PATH`. If you prefer managed Python versions, set up [pyenv](https://github.com/pyenv/pyenv) and run `pyenv install 3.9.XX` followed by `pyenv local 3.9.XX` inside the repo.
2. Create a virtual environment: `python3.9 -m venv venv`
3. Activate it:
   - Linux/macOS: `source ./venv/bin/activate`
   - Windows (PowerShell): `./venv/Scripts/Activate.ps1`
4. Install the core Python dependencies: `pip install -r requirements.txt`
5. Install hardware-optimised runtimes (pick what suits your machine):
   - PyTorch: follow the command generator at <https://pytorch.org/get-started/locally/>
   - ONNX Runtime: follow <https://onnxruntime.ai/docs/install/> for your accelerator
6. Install Ultralytics (YOLO backend): `pip install ultralytics`
7. Verify that your environment can see the available devices: `python devices.py`

Once the prerequisites are satisfied you can configure the server, load a model, and start handling requests.

Prerequisites
-------------

- Python 3.9 (other versions are untested)
- System-level GPU/NPU drivers that match the accelerator you plan to target (CUDA, Intel GPU/NPU, etc.)
- (Optional) Docker with access to the required device runtime for containerised deployments

Environment Setup
-----------------

Create and activate the virtual environment as shown in the quick-start section. After activation:

```bash
# Install Python packages that do not depend on hardware
pip install -r requirements.txt

# Install Ultralytics YOLO runtime
pip install ultralytics

# Install accelerator-specific stacks separately.
# Examples (consult the linked docs for an exact command):
#   PyTorch with CUDA: https://pytorch.org/get-started/locally/
#   ONNX Runtime variants: https://onnxruntime.ai/docs/install/
```

Manual installation of CUDA-enabled PyTorch inside the activated virtual environment tends to be more reliable than relying on `environment.yml`, as the installers expect an active environment.

After installing runtimes, validate the detected devices:

```bash
python devices.py
```

Configuration
-------------

1. Copy the template configuration: `cp example-config.ini config.ini`
2. Update the `[InferenceServer]` section with your desired host, port, and model list. At minimum provide at least one model path, for example:
   ```
   [InferenceServer]
   Host=0.0.0.0
   Port=50051
   MaxWorkers=1
   Device=cuda:0     ; or intel:gpu / cpu / etc.
   Models=/absolute/path/to/model.onnx
   ```
3. Review the rest of the configuration file to make sure any detector- or environment-specific settings match your deployment.

Generating Protobuf Files
--------------------------

If you modify the `.proto` definition, regenerate the Python and Go bindings:

**Python:**
```bash
python -m grpc_tools.protoc \
  -I./protos \
  --python_out=. \
  --grpc_python_out=. \
  --pyi_out=. \
  ./protos/detect_object.proto
```

**Go:**
```bash
protoc \
  -I./protos \
  --go_out=./go_output \
  --go_opt=paths=source_relative \
  --go-grpc_out=./go_output \
  --go-grpc_opt=paths=source_relative \
  ./protos/detect_object.proto
```

The Python command generates files in the repository root (`detect_object_pb2.py`, `detect_object_pb2_grpc.py`, `detect_object_pb2.pyi`). The Go command outputs to `go_output/protos/`.

Running the Server Locally
--------------------------

1. Place the ONNX model(s) referenced in `config.ini` either in the repository `model/` directory or at another path accessible from the machine.
2. (Optional) Populate an `input/` directory with PNG images if you intend to run the included test client or your own scripts.
3. Start the gRPC server:
   ```bash
   python server.py config.ini
   ```
   On startup the server will load the configured model, warm it up on the selected device, and begin listening for requests.

Manual Testing
--------------

- Use the gRPC test client (`test_client.py`) or your own client implementation to send images to the server. The script should iterate through each PNG in the `./input` directory, invoke the inference endpoint, and log performance metrics.
- Customise the client-side confidence and IoU thresholds as needed to match production settings.
- If you do not already have `test_client.py`, adapt one of the gRPC samples or request the internal script used by the team.

Profiling the Server
--------------------

Python's built-in profiler and SnakeViz can help diagnose performance hotspots:

```bash
python -m cProfile -o server_profile.prof server.py config.ini

# Visualise the profile
pip install snakeviz
snakeviz server_profile.prof
```

OpenVINO Runtime (Intel Accelerators)
-------------------------------------

Follow the official documentation to install OpenVINO:

- Base installation: <https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-apt.html>
- Additional GPU configuration: <https://docs.openvino.ai/2025/get-started/install-openvino/configurations/configurations-intel-gpu.html>

Install the extra system dependencies:

```bash
sudo apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero
```

Depending on your system you may also need Intel driver libraries from <https://dgpu-docs.intel.com/driver/installation.html#ubuntu#ubuntu>. If the package repository fails to install, see the troubleshooting advice at <https://github.com/intel/intel-extension-for-pytorch/issues/365>.

When selecting a device in configuration or Ultralytics commands you can use `intel:gpu`, `intel:npu`, or `intel:cpu`.

Prebuilt Docker Images
----------------------

If you only need to run the service, pull one of the published images instead of building locally:

- `docker pull redfoxid/inference-server:production`
- `docker pull redfoxid/inference-server:openvino-cpu`
- `docker pull redfoxid/inference-server:openvino-gpu`
- `docker pull redfoxid/inference-server:cuda`

Docker Workflow
---------------

### Build the image

```bash
docker build -t redfoxid/inference-server .
```

Mount your model directory read-only so the container can access the ONNX file.

### Build multiple runtimes with Buildx

If you have Docker Buildx available, the repo includes a `docker-bake.hcl` that maps to every published runtime tag:

```bash
# Build them all (production, cuda, openvino-cpu/gpu/npu)
docker buildx bake all

# Build just one target
docker buildx bake cuda

# Override the CUDA wheel tag when building
docker buildx bake cuda --set cuda.args.TORCH_CUDA_TAG=cu122
```

Set the environment variables `REGISTRY` or `BUILD_CONTEXT` to override the default values used in the bake file (`redfoxid/inference-server` and current directory respectively).

### Verify hardware from inside the container

If hardware support looks uncertain, run the bundled hardware probe from within the container before starting the server. Swap the image tag and device options to match the runtime you plan to test.

```bash
docker run --rm \
  -v "$(pwd)/config.ini:/app/config.ini:ro" \
  -v "$(pwd)/model:/app/model:ro" \
  --gpus all \
  redfoxid/inference-server:cuda \
  python devices.py
```

For OpenVINO variants replace the tag (for example `openvino-gpu`) and substitute `--gpus all` with either `--device=/dev/dri` or `--device=/dev/accel` depending on GPU versus NPU access.

### NVIDIA GPUs

1. Install the NVIDIA container toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
2. Run the container with GPU support:
   ```bash
   docker run -v "$(pwd)/model:/app/model/:ro" \
              -v "$(pwd)/config.ini:/app/config.ini:ro" \
              --gpus all \
              --rm \
              -p 50051:50051 \
              --name inference-server \
              redfoxid/inference-server:cuda config.ini
   ```
   Omit `--rm` if you prefer to retain the container between runs.

### Intel GPUs / NPUs (OpenVINO)

Use the appropriate device path instead of `--gpus all`:

- Intel GPU: `--device=/dev/dri`
- Intel NPU: `--device=/dev/accel`

Example:

For Intel GPU workloads use the OpenVINO GPU image:

```bash
docker run -v "$(pwd)/model:/app/model/:ro" \
           -v "$(pwd)/config.ini:/app/config.ini:ro" \
           --device=/dev/dri \
           -d \
           -p 50051:50051 \
           --name inference-server \
           redfoxid/inference-server:openvino-gpu config.ini
```

For Intel NPU workloads change the device flag to `/dev/accel` while keeping the OpenVINO GPU tag:

```bash
docker run -v "$(pwd)/model:/app/model/:ro" \
           -v "$(pwd)/config.ini:/app/config.ini:ro" \
           --device=/dev/accel \
           -d \
           -p 50051:50051 \
           --name inference-server \
           redfoxid/inference-server:openvino-npu config.ini
```

The OpenVINO GPU image now follows Intel's client driver guidance and installs the dGPU user-space components (OpenCL ICD, Level Zero userspace, metrics discovery, VA/Media drivers, `intel-gsc`, `libvpl*`, `clinfo`, `vainfo`, `intel-ocloc`, etc.). You still need the matching Intel kernel driver stack on the host and must pass `/dev/dri` through to the container.

The OpenVINO NPU image bundles Intel's `linux-npu-driver` release `v1.24.0` (Ubuntu 24.04 tarball), Level Zero loader `1.24.2`, and the required USB/runtime dependencies (`libusb-1.0-0`, `udev`, `libtbb12`). Ensure the host exposes the VPU device path (commonly `/dev/accel` or the relevant `/dev/bus/usb` device) with the `--device` flag so OpenVINO can detect it inside the container.

For CPU-only execution swap the device flag and tag:

```bash
docker run -v "$(pwd)/model:/app/model/:ro" \
           -v "$(pwd)/config.ini:/app/config.ini:ro" \
           -d \
           -p 50051:50051 \
           --name inference-server \
           redfoxid/inference-server:openvino-cpu config.ini
```

Troubleshooting Tips
--------------------

- If models fail to load, double-check the paths listed under `Models=` in `config.ini`. Absolute paths avoid ambiguity when running inside Docker.
- If no accelerator is detected, rerun `python devices.py` after confirming the relevant drivers and runtimes are installed.
- Slow first-inference times are expected; subsequent requests should be faster after the model warms up.

Additional Resources
--------------------

- PyTorch installation helper: <https://pytorch.org/get-started/locally/>
- ONNX Runtime installation guide: <https://onnxruntime.ai/docs/install/>
- OpenVINO installation guide: <https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-apt.html>
