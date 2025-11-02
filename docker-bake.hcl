variable "build_context" {
  default = "."
}

variable "registry" {
  default = "redfoxid/inference-server"
}

variable "torch_cuda_tag" {
  default = "cu126"
}

# Maintain manually if torch_cuda_tag changes
variable "cuda_major_minor" {
  default = "12.6"
}

# Full TensorRT version used for pip install
variable "tensorrt_version" {
  default = "10.13.3.9"
}

# Maintain manually alongside tensorrt_version
variable "tensorrt_major_minor" {
  default = "10.13"
}

target "base" {
  context    = build_context
}

target "production" {
  inherits = ["base"]
  target   = "production"
  tags     = ["${registry}:production"]
}

target "cuda" {
  inherits = ["base"]
  target   = "cuda"
  args = {
    TORCH_CUDA_TAG = torch_cuda_tag
  }
  tags = ["${registry}:cuda${cuda_major_minor}"]
}

target "trt" {
  inherits = ["base"]
  target   = "trt"
  args = {
    TORCH_CUDA_TAG = torch_cuda_tag
    TENSORRT_VERSION = tensorrt_version
  }
  tags = [
    "${registry}:cuda${cuda_major_minor}-trt${tensorrt_major_minor}",
  ]
}

target "openvino-base" {
  inherits = ["base"]
  target   = "openvino-base"
}

target "openvino-cpu" {
  inherits = ["openvino-base"]
  target   = "openvino-cpu"
  tags     = ["${registry}:openvino-cpu"]
}

target "openvino-gpu" {
  inherits = ["openvino-base"]
  target   = "openvino-gpu"
  tags     = ["${registry}:openvino-gpu"]
}

target "openvino-npu" {
  inherits = ["openvino-base"]
  target   = "openvino-npu"
  tags     = ["${registry}:openvino-npu"]
}

group "all" {
  targets = [
    "production",
    "openvino-cpu",
    "openvino-gpu",
    "openvino-npu",
  ]
}
