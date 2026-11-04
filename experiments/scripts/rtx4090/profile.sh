#!/usr/bin/env bash
# Short, matched baseline and Nsight runs on the existing four-GPU deployment.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
cd "$RTX4090_REPO_DIR"

mode="${1:-trace}"
if (( $# > 0 )); then shift; fi
case "$mode" in
  baseline|trace|cpu) ;;
  *) echo "Usage: $0 {baseline|trace|cpu} [--dry-run] [benchmark options]" >&2; exit 2 ;;
esac
dry_run=0
for arg in "$@"; do
  case "$arg" in --dry-run|--help|-h) dry_run=1 ;; esac
  case "$arg" in
    --nsys|--profile-dir|--profile-dir=*|-p|--trace|--enable-trace-detail|--enable-advanced-logging)
      echo "profile.sh chooses one collector; omit $arg and other profiling flags." >&2; exit 2 ;;
    --file|--file=*|-f)
      echo 'profile.sh records metrics in its run directory; set PROFILE_RUN_DIR instead of --file.' >&2; exit 2 ;;
  esac
done

run_dir="${PROFILE_RUN_DIR:-$RTX4090_REPO_DIR/reports/rtx4090/profile-$mode-$(date +%Y%m%d-%H%M%S)-$$}"
export DMOE_PROFILE_OUTPUT_DIR="$run_dir/nsight"
# Keep the legacy in-memory recorder disabled; the new bridge emits real NVTX.
export ENABLE_NVTX=0 DMOE_ENABLE_NVTX=0 DMOE_NSYS_CPU_SAMPLING=0
if [[ "$mode" != baseline ]]; then
  export DMOE_ENABLE_NVTX=1
fi
if [[ "$mode" == cpu ]]; then
  export DMOE_NSYS_CPU_SAMPLING=1
fi

if (( ! dry_run )); then
  if [[ "$mode" != baseline ]]; then
    command -v nsys >/dev/null || { echo 'Install Nsight Systems CLI first; see docs/rtx4090-profiling.md.' >&2; exit 1; }
  fi
  [[ ! -e "$run_dir" ]] || { echo "Choose a new PROFILE_RUN_DIR; $run_dir already exists." >&2; exit 1; }
  mkdir -p "$run_dir"
  if [[ "$mode" != baseline ]]; then
    nsys --version > "$run_dir/nsys-version.txt"
    # Failed CPU sampling diagnostics do not prohibit CUDA/NVTX tracing.
    nsys status --environment > "$run_dir/nsys-environment.txt" 2>&1 || true
  fi
fi

args=(benchmark --num-requests 8 --rate 1 --file "$run_dir/metrics.csv")
if [[ "$mode" != baseline ]]; then args+=(--nsys); fi
args+=("$@")
echo "Profile mode: $mode; run directory: $run_dir"
if (( dry_run )); then
  bash "$RTX4090_SCRIPT_DIR/launch.sh" "${args[@]}"
else
  bash "$RTX4090_SCRIPT_DIR/launch.sh" "${args[@]}" 2>&1 | tee "$run_dir/run.log"
fi
