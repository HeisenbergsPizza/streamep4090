#!/usr/bin/env bash
# Source from the other scripts; paths are relative to this checkout.
RTX4090_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RTX4090_REPO_DIR="$(cd -- "$RTX4090_SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
RTX4090_PARENT_TMPDIR="${TMPDIR:-/tmp}"

rtx4090_env() {
  export ENABLE_GDRCOPY=0 DMOE_USE_GDRCOPY=0 KERNEL_USE_GDRCOPY=0 ENABLE_NIXL=0
  export TORCH_CUDA_ARCH_LIST=8.9
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export VLLM_USE_V1=0 VLLM_FLASH_ATTN_VERSION=2
  export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=0
  export NCCL_CUMEM_HOST_ENABLE=0
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$RTX4090_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
  # C++ TEMP_DIR is a build-time macro; use the same path at build and runtime.
  export TMPDIR="${DMOE_TMPDIR:-/tmp/streaminfer-rtx4090/}"
  export TMPDIR="${TMPDIR%/}/"
}

rtx4090_libraries() {
  # Prefer the NCCL headers/library shipped with the installed torch stack.
  local nccl_root torch_lib
  nccl_root="$("$PYTHON" -c 'import importlib.metadata as m; print(m.distribution("nvidia-nccl-cu12").locate_file("nvidia/nccl"))')"
  torch_lib="$("$PYTHON" -c 'import importlib.metadata as m; print(m.distribution("torch").locate_file("torch/lib"))')"
  export NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-$nccl_root/include}"
  export NCCL_LIBRARY_DIR="${NCCL_LIBRARY_DIR:-$nccl_root/lib}"
  export LD_LIBRARY_PATH="$NCCL_LIBRARY_DIR:$torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LIBRARY_PATH="$NCCL_LIBRARY_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
  # Some NVIDIA Python wheels ship only libnccl.so.2. -lnccl needs .so.
  if [[ ! -f "$NCCL_LIBRARY_DIR/libnccl.so" && -f "$NCCL_LIBRARY_DIR/libnccl.so.2" ]]; then
    local linker_dir="$RTX4090_REPO_DIR/build/rtx4090-link"
    mkdir -p "$linker_dir"
    ln -sfn -- "$NCCL_LIBRARY_DIR/libnccl.so.2" "$linker_dir/libnccl.so"
    export LIBRARY_PATH="$linker_dir:$LIBRARY_PATH"
  fi
}

rtx4090_print_command() {
  printf '%q ' "$@"
  printf '\n'
}
