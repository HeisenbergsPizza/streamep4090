#!/usr/bin/env bash
# Qwen-shaped synthetic decode on 4x4090: 512 requests, DP4/EP4/TP1.
# Attention and experts both use CUDA Graphs. Defaults: KV length 64,
# 128 output tokens, attention cap 128, expert-branch cap 128, defrag.
#
# Usage (activate the streamep4090 conda environment first):
#   bash experiments/scripts/rtx4090/profile_graph_b512.sh
#   bash experiments/scripts/rtx4090/profile_graph_b512.sh --dry-run
#   bash experiments/scripts/rtx4090/profile_graph_b512.sh baseline
# Extra benchmark options override these defaults, as in profile.sh.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
mode=trace
if (( $# > 0 )); then
  case "$1" in
    trace|baseline|cpu) mode="$1"; shift ;;
  esac
fi

# Keep graph reports identifiable and allow extra time for graph capture.
export PROFILE_RUN_DIR="${PROFILE_RUN_DIR:-$repo_dir/reports/rtx4090/profile-$mode-cudagraph-b512-$(date +%Y%m%d-%H%M%S)-$$}"
export RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-1800}"

exec bash "$script_dir/profile.sh" "$mode" \
  --synchronized-start \
  --generator-type offline \
  --num-requests 512 \
  --max-batch-size-attn 128 \
  --max-attn-graph-bsz 128 \
  --max-batch-size-expert 128 \
  --cuda-graph-attn \
  --cuda-graph-expert \
  --unified-scheduler-type defrag \
  --min-input-len 64 --max-input-len 65 \
  --min-output-len 128 --max-output-len 129 \
  "$@"
