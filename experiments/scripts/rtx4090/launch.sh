#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
cd "$RTX4090_REPO_DIR"
rtx4090_env
mode="${1:-smoke}"
if (( $# > 0 )); then shift; fi
case "$mode" in smoke|benchmark|server) ;; *) echo "Usage: $0 {smoke|benchmark|server} [--dry-run] [benchmark CLI options]" >&2; exit 2 ;; esac
for arg in "$@"; do
  if [[ "$arg" == --dry-run || "$arg" == --help || "$arg" == -h ]]; then
    export TMPDIR="$RTX4090_PARENT_TMPDIR"
    exec "$PYTHON" "$RTX4090_SCRIPT_DIR/run_local.py" "$mode" "$@"
  fi
done
rtx4090_libraries
mkdir -p "$TMPDIR" "$RTX4090_REPO_DIR/reports/rtx4090"
log="$RTX4090_REPO_DIR/reports/rtx4090/$mode-$(date +%Y%m%d-%H%M%S).log"
echo "Log: $log"
echo 'NCCL: P2P disabled, IB disabled, SHM enabled. No-GDR / BF16 / sm89.'
if [[ "$mode" == server ]]; then
  "$PYTHON" "$RTX4090_SCRIPT_DIR/run_local.py" "$mode" "$@" 2>&1 | tee "$log"
else
  command -v timeout >/dev/null
  timeout --signal=TERM --kill-after=30 "${RUN_TIMEOUT_SECONDS:-900}" \
    "$PYTHON" "$RTX4090_SCRIPT_DIR/run_local.py" "$mode" "$@" 2>&1 | tee "$log"
fi
