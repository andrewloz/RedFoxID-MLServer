variable "build_context" {
  default = "."
}

variable "registry" {
  default = "redfoxid/inference-server"
}

variable "torch_cuda_tag" {
  default = "cu126"
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
  tags = ["${registry}:cuda"]
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
