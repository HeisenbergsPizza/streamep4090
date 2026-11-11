#!/usr/bin/env bash
# Validate/time one local MoE, then collect one selected stage with Nsight Compute.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash profile_experts.sh BACKEND STAGE --case PATH [options]

  BACKEND                streaminfer | sglang (use its existing Python environment)
  STAGE                  up | activation | down | full
  --case PATH            Shared case created by expert_bench.py prepare
  --output-dir PATH      New report directory; existing directories are rejected
  --ncu-bin PATH         Nsight Compute executable (otherwise $NCU_BIN or ncu)
  --detailed             Add compute and memory analysis to basic counter sections
  --dry-run              Print both commands without creating files or using a GPU

Environment:
  PYTHON_BIN             Python executable; default: python
  CUDA_VISIBLE_DEVICES   One idle physical GPU; default: 0
  EXPERT_REPORT_ROOT     Default parent directory for unique reports

Ordinary timing measures each backend's native forward, even for a selected stage.
The two native forwards have different reordering/combine boundaries; see the guide.
The NCU pass profiles only the selected NVTX range after warmup. No NCCL or nsys.
USAGE
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
print_command() { printf ' '; printf ' %q' "$@"; printf '\n'; }

if (( $# == 0 )) || [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  usage
  exit 0
fi
(( $# >= 2 )) || die 'BACKEND and STAGE are required; use --help.'
backend="$1"
stage="$2"
shift 2
case "$backend" in streaminfer|sglang) ;; *) die "Unknown backend: $backend" ;; esac
case "$stage" in up|activation|down|full) ;; *) die "Unknown stage: $stage" ;; esac

case_path=''
output_dir=''
ncu_bin="${NCU_BIN:-ncu}"
python_bin="${PYTHON_BIN:-python}"
detailed=0
dry_run=0
while (( $# > 0 )); do
  case "$1" in
    --case|--output-dir|--ncu-bin)
      (( $# >= 2 )) || die "Missing value for $1"
      [[ -n "$2" && "$2" != --* ]] || die "Missing value for $1"
      case "$1" in
        --case) case_path="$2" ;;
        --output-dir) output_dir="$2" ;;
        --ncu-bin) ncu_bin="$2" ;;
      esac
      shift 2
      ;;
    --detailed) detailed=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done
[[ -n "$case_path" ]] || die '--case is required.'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
bench="$script_dir/expert_bench.py"
if [[ -z "$output_dir" ]]; then
  output_dir="${EXPERT_REPORT_ROOT:-$repo_dir/reports/expert-kernels}/$backend-$stage-$(date +%Y%m%d-%H%M%S)-$$"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
[[ -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || \
  die 'Select exactly one GPU with CUDA_VISIBLE_DEVICES, e.g. CUDA_VISIBLE_DEVICES=0.'

plain=("$python_bin" "$bench" run --backend "$backend" --stage "$stage"
  --case "$case_path" --output "$output_dir/timing.json")
capture=("$ncu_bin" --nvtx --nvtx-include "ExpertBench.$stage/"
  --profile-from-start off --replay-mode kernel --kernel-name-base demangled
  --clock-control none --cache-control all
  --section SpeedOfLight --section LaunchStats --section Occupancy)
if (( detailed )); then
  capture+=(--section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis)
fi
capture+=(-o "$output_dir/kernels" "$python_bin" "$bench" run
  --backend "$backend" --stage "$stage" --case "$case_path"
  --output "$output_dir/capture.json" --profile-only)

printf 'Backend: %s; stage: %s; CUDA_VISIBLE_DEVICES=%s\n' "$backend" "$stage" "$CUDA_VISIBLE_DEVICES"
printf 'Output: %s\n' "$output_dir"
printf 'Validate and time (without NCU):\n'
print_command "${plain[@]}"
printf 'Capture selected stage:\n'
print_command "${capture[@]}"
(( dry_run )) && exit 0

[[ -f "$bench" ]] || die "Missing benchmark: $bench"
[[ -f "$case_path" ]] || die "Missing case: $case_path"
command -v "$python_bin" >/dev/null 2>&1 || die "Python not found: $python_bin"
command -v "$ncu_bin" >/dev/null 2>&1 || die "NCU not found: $ncu_bin; set NCU_BIN to its absolute path."
[[ ! -e "$output_dir" ]] || die "Output already exists: $output_dir; select a new directory."
# mkdir without -p for the final component prevents concurrent runs sharing a report.
mkdir -p -- "$(dirname -- "$output_dir")"
mkdir -- "$output_dir"
{
  printf 'CUDA_VISIBLE_DEVICES=%q\n' "$CUDA_VISIBLE_DEVICES"
  print_command "${plain[@]}"
  print_command "${capture[@]}"
} > "$output_dir/commands.txt"
"$ncu_bin" --version | tee "$output_dir/ncu-version.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$output_dir/nvidia-smi-before.txt" || true
fi

"${plain[@]}" 2>&1 | tee "$output_dir/timing.log"
"${capture[@]}" 2>&1 | tee "$output_dir/capture.log"
[[ -s "$output_dir/kernels.ncu-rep" ]] || \
  die 'NCU did not write a report. Inspect capture.log for permission, compatibility, or filter errors.'
"$ncu_bin" --import "$output_dir/kernels.ncu-rep" --page details > "$output_dir/details.txt"
printf 'Completed. Timing: %s/timing.json\nReport: %s/kernels.ncu-rep\n' "$output_dir" "$output_dir"
