#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
cd "$RTX4090_REPO_DIR"
rtx4090_env

dry_run=0
system_deps=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    --system-deps) system_deps=1 ;;
    *) echo "Usage: $0 [--dry-run] [--system-deps]" >&2; exit 2 ;;
  esac
done
run() { rtx4090_print_command "$@"; if (( ! dry_run )); then "$@"; fi; }

echo 'Build: ENABLE_GDRCOPY=0 DMOE_USE_GDRCOPY=0 KERNEL_USE_GDRCOPY=0 ENABLE_NIXL=0 TORCH_CUDA_ARCH_LIST=8.9'
if (( system_deps )); then
  run sudo apt-get update
  run sudo apt-get install -y build-essential git pkg-config libzmq3-dev libucx-dev
  # Ubuntu 22.04 bundles zmq.hpp in libzmq3-dev; 24.04 splits it out.
  if (( dry_run )); then
    echo 'If /usr/include/zmq.hpp is missing after installing libzmq3-dev: sudo apt-get install -y cppzmq-dev'
  elif [[ ! -f /usr/include/zmq.hpp ]]; then
    run sudo apt-get install -y cppzmq-dev
  fi
fi
if (( ! dry_run )); then
  "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Activate a dedicated Python 3.12 environment first"; assert sys.prefix != sys.base_prefix or "conda" in sys.version.lower() or __import__("os").environ.get("CONDA_PREFIX"), "Use a venv or conda environment"'
  command -v nvcc >/dev/null || { echo 'Set CUDA_HOME and put its bin directory on PATH (CUDA Toolkit required).' >&2; exit 1; }
  export CUDA_HOME="${CUDA_HOME:-$(dirname -- "$(dirname -- "$(command -v nvcc)")")}"
  export ZMQ_HOME="${ZMQ_HOME:-/usr}"
  for header in /usr/include/zmq.h /usr/include/zmq.hpp /usr/include/ucp/api/ucp.h; do
    [[ -f "$header" ]] || { echo "Missing $header; install the system packages listed in docs/rtx4090.md (or use --system-deps)." >&2; exit 1; }
  done
  mkdir -p "$TMPDIR"
fi

# Only initialize build dependencies; the separate uccl benchmark is unnecessary.
if [[ -e .git ]]; then
  run git submodule update --init --recursive third_party/cereal third_party/NVTX third_party/pybind11 third_party/cutlass
fi
run "$PYTHON" -m pip install 'pip<26' 'setuptools<80' wheel ninja
run "$PYTHON" -m pip install -c "$RTX4090_SCRIPT_DIR/constraints.txt" torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# vLLM 0.8.2 pins a release removed from PyPI: https://github.com/mlc-ai/xgrammar/issues/804
xgrammar_version=""
if (( dry_run )); then
  echo 'If xgrammar 0.1.16 is not installed in this Python environment, build the pinned upstream source:'
else
  xgrammar_version="$("$PYTHON" - <<'PY'
import importlib.metadata as m
try:
    print(m.version("xgrammar"))
except m.PackageNotFoundError:
    pass
PY
)"
fi
if [[ "$xgrammar_version" == "0.1.16" ]]; then
  echo 'xgrammar 0.1.16 already installed; skipping source build.'
else
  run "$PYTHON" -m pip install 'cmake>=3.18,<4' 'scikit-build-core>=0.10,<0.11' pybind11==2.13.6
  # Bootstrap only this package; requirements below resolve its runtime dependencies.
  run env "CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS:-4}" "$PYTHON" -m pip install --no-deps --no-build-isolation \
    'xgrammar @ git+https://github.com/mlc-ai/xgrammar.git@989222175c2a30fb7987d8bcce35bec1bf6817f2'
  run "$PYTHON" -c 'import importlib.metadata as m; version = m.version("xgrammar"); assert version == "0.1.16", f"Expected xgrammar 0.1.16, got {version}"'
fi

run "$PYTHON" -m pip install -c "$RTX4090_SCRIPT_DIR/constraints.txt" -r requirements.txt 'flask[async]' pytest
# Package metadata alone does not catch Arrow/NumPy binary incompatibility.
run "$PYTHON" -c 'import numpy, pandas, pyarrow, pyarrow.parquet; print("Data dependencies:", numpy.__version__, pandas.__version__, pyarrow.__version__)'
if (( dry_run )); then
  echo 'Apply patches/vllm_0.8.2.patch to the installed vLLM 0.8.2 package (idempotent check).'
  echo 'Detect installed torch/NCCL library paths; verify CUDA headers and submodules.'
  rtx4090_print_command "$PYTHON" -m pip install -e . --no-build-isolation --no-deps
  exit 0
fi

"$PYTHON" - <<'PY'
import importlib.metadata as m
import os
from pathlib import Path
import re
import subprocess
assert m.version("vllm") == "0.8.2", "Expected vLLM 0.8.2"
site_dir = Path(m.distribution("vllm").locate_file("")).resolve()
patch = str(Path("patches/vllm_0.8.2.patch").resolve())
base = ["git", "-C", str(site_dir), "apply"]
if subprocess.run(base + ["--reverse", "--check", patch], capture_output=True).returncode == 0:
    print("vLLM patch already applied")
else:
    subprocess.run(base + ["--check", patch], check=True)
    subprocess.run(base + [patch], check=True)
for relative in ("third_party/cereal/include/cereal/cereal.hpp", "third_party/cutlass/include/cutlass/cutlass.h", "third_party/NVTX/c/include/nvtx3/nvToolsExt.h", "third_party/pybind11/include/pybind11/pybind11.h"):
    assert Path(relative).is_file(), f"Missing submodule content: {relative}"
import torch
assert torch.__version__.split("+")[0] == "2.6.0"
print("torch:", torch.__version__, "CUDA wheel:", torch.version.cuda)
nvcc = Path(os.environ["CUDA_HOME"]) / "bin/nvcc"
nvcc_output = subprocess.check_output([str(nvcc), "--version"], text=True)
print(nvcc_output)
toolkit = re.search(r"release (\d+)\.(\d+)", nvcc_output)
assert toolkit and torch.version.cuda, "Unable to determine CUDA toolkit/wheel versions"
toolkit_version = tuple(map(int, toolkit.groups()))
wheel_version = tuple(map(int, torch.version.cuda.split(".")[:2]))
assert toolkit_version[0] == wheel_version[0], f"CUDA major mismatch: toolkit {toolkit_version}, torch wheel {wheel_version}"
if toolkit_version != wheel_version:
    print(f"WARNING: toolkit {toolkit_version} differs from torch CUDA {wheel_version}; align versions if extension compilation or import fails.")
PY
rtx4090_libraries
[[ -f "$NCCL_INCLUDE_DIR/nccl.h" ]] || { echo "Missing $NCCL_INCLUDE_DIR/nccl.h" >&2; exit 1; }
export MAX_JOBS="${MAX_JOBS:-4}"
run "$PYTHON" -m pip install -e . --no-build-isolation --no-deps
run "$PYTHON" -m pip check
mkdir -p reports/rtx4090
"$PYTHON" -m pip freeze > reports/rtx4090/installed-packages.txt
"$PYTHON" -c 'import torch, disagmoe_c as c; assert not c.gdrcopy_compiled(); assert not c.gdrcopy_enabled(); print("Extension import and no-GDR build checks passed; GPU preflight remains required.")'
