#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
cd "$RTX4090_REPO_DIR"
rtx4090_env
if [[ "${1:-}" == --dry-run ]]; then
  echo 'Environment: no GDRCopy, no NIXL, NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=0 CUDA_VISIBLE_DEVICES='"$CUDA_VISIBLE_DEVICES"
  rtx4090_print_command nvidia-smi topo -m
  rtx4090_print_command timeout --signal=TERM --kill-after=15 120 "$PYTHON" "$RTX4090_SCRIPT_DIR/preflight.py"
  rtx4090_print_command timeout --signal=TERM --kill-after=15 180 "$PYTHON" -m pytest -q tests/ops/test_host_staging.py tests/ops/test_token_routing.py tests/engine/test_shutdown.py
  rtx4090_print_command timeout --signal=TERM --kill-after=15 180 "$PYTHON" tests/ops/test_grouped_gemm.py
  rtx4090_print_command timeout --signal=TERM --kill-after=15 180 "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 "$RTX4090_SCRIPT_DIR/nccl_pairwise.py"
  exit 0
fi
[[ $# == 0 ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }
rtx4090_libraries
command -v timeout >/dev/null
mkdir -p "$TMPDIR" "$RTX4090_REPO_DIR/reports/rtx4090"
log="$RTX4090_REPO_DIR/reports/rtx4090/preflight-$(date +%Y%m%d-%H%M%S).log"
{
  nvidia-smi
  nvidia-smi topo -m
  timeout --signal=TERM --kill-after=15 120 "$PYTHON" "$RTX4090_SCRIPT_DIR/preflight.py"
  timeout --signal=TERM --kill-after=15 180 "$PYTHON" -m pytest -q tests/ops/test_host_staging.py tests/ops/test_token_routing.py tests/engine/test_shutdown.py
  timeout --signal=TERM --kill-after=15 180 "$PYTHON" tests/ops/test_grouped_gemm.py
  timeout --signal=TERM --kill-after=15 180 "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 "$RTX4090_SCRIPT_DIR/nccl_pairwise.py"
} 2>&1 | tee "$log"
echo "Preflight passed; log: $log. Next run the 12-layer smoke test."
