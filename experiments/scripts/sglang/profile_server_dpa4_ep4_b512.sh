#!/usr/bin/env bash
# Qwen3-30B-A3B: 512 total requests, Attention DP4/TP1, Experts EP4/TP1.
# The vocabulary LM head retains SGLang's default TP4. No DeepEP is used.
# Set SGLANG_DISABLE_CUDA_GRAPH=1 in both server and client commands for eager.
set -eo pipefail

mode="${1:-run}"
if [[ $# -gt 1 || ( "$mode" != run && "$mode" != --check-only ) ]]; then
  echo "Usage: bash $0 [--check-only]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
script_path="$script_dir/$(basename -- "${BASH_SOURCE[0]}")"
source "${SGLANG_WORK:-/workspace/zzc/sglang-qwen30b}/activate.sh"

export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-0}"
case "$SGLANG_DISABLE_CUDA_GRAPH" in
  0) graph_mode=graph-on ;;
  1) graph_mode=graph-off ;;
  *) echo "SGLANG_DISABLE_CUDA_GRAPH must be 0 or 1" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
export SGLANG_PROFILE_V2=0

server_args=(
  --model-path "$MODEL_DIR"
  --served-model-name Qwen3-30B-A3B
  --host 127.0.0.1 --port 30000
  --tp-size 4 --dp-size 4 --enable-dp-attention --ep-size 4
  --load-balance-method round_robin
  --dtype bfloat16
  --attention-backend triton
  --moe-a2a-backend none --moe-runner-backend triton
  --sampling-backend pytorch
  --disable-custom-all-reduce
  --mem-fraction-static 0.80
  --context-length 2048
  # SGLang divides this global prefill budget by DP4, giving 512 per DP.
  --chunked-prefill-size 2048
  # Global request limit; the per-DP limit and graph width are 128.
  --max-running-requests 512 --cuda-graph-max-bs 128
  --enable-layerwise-nvtx-marker
)
if [[ "$SGLANG_DISABLE_CUDA_GRAPH" == 1 ]]; then
  server_args+=(--disable-cuda-graph)
fi

check_config() {
  python -B - "${server_args[@]}" <<'PY'
import json
import os
import sys
from sglang.srt.server_args import prepare_server_args

args = prepare_server_args(sys.argv[1:])
args.check_server_args()
expected = {
    "tp_size": 4, "dp_size": 4, "ep_size": 4, "moe_dp_size": 1,
    "attn_cp_size": 1, "enable_dp_attention": True,
    "enable_dp_lm_head": False, "max_running_requests": 512,
    "cuda_graph_max_bs": 128,
    "disable_cuda_graph": os.environ["SGLANG_DISABLE_CUDA_GRAPH"] == "1",
    "moe_a2a_backend": "none", "moe_runner_backend": "triton",
    "load_balance_method": "round_robin", "chunked_prefill_size": 512,
}
for key, value in expected.items():
    actual = getattr(args, key)
    if actual != value:
        raise ValueError(f"{key}: expected {value!r}, got {actual!r}")
print(json.dumps({
    "configuration": {key: getattr(args, key) for key in expected},
    "attention_tp_size": args.tp_size // args.dp_size // args.attn_cp_size,
    "expert_tp_size": args.tp_size // args.ep_size // args.moe_dp_size,
    "requested_max_running_requests_per_dp": args.max_running_requests // args.dp_size,
    "schedule_conservativeness": args.schedule_conservativeness,
    "note": "Configuration check only; no weights loaded or inference started.",
}, indent=2))
PY
}

if [[ "$mode" == --check-only ]]; then
  check_config
  exit 0
fi

command -v nsys >/dev/null
# Refuse to start beside an older service on this port; never stop it here.
python -B - <<'PY'
import socket
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 30000))
PY

report_dir="$SGLANG_WORK/reports/$(date +%Y%m%d-%H%M%S)-dpa4-ep4-b512-$graph_mode-$$"
mkdir -p "$report_dir"
check_config > "$report_dir/config-validation.json"
python -m pip freeze > "$report_dir/pip-freeze.txt"
nsys --version > "$report_dir/nsys-version.txt"
cp "$MODEL_DIR/config.json" "$report_dir/model-config.json"
if [[ -f "$MODEL_DIR/download_revision.txt" ]]; then
  cp "$MODEL_DIR/download_revision.txt" "$report_dir/model-revision.txt"
fi
cp "$script_path" "$report_dir/profile_server.sh"
cp "$script_dir/profile_client_dpa4_ep4_b512.py" "$report_dir/profile_client.py"
printf '%q ' python -m sglang.launch_server "${server_args[@]}" > "$report_dir/command.txt"
printf '\n' >> "$report_dir/command.txt"
echo "本次报告目录：$report_dir"
echo "全局请求上限 512；Attention DP4/TP1；Experts EP4/TP1；CUDA Graph: $graph_mode。"
if [[ "$SGLANG_DISABLE_CUDA_GRAPH" == 0 ]]; then
  echo "每 DP Graph 上限 128。"
fi
echo "服务就绪后，在另一终端运行 profile_client_dpa4_ep4_b512.py。"
echo "客户端也需设置 SGLANG_DISABLE_CUDA_GRAPH=$SGLANG_DISABLE_CUDA_GRAPH。"

cd "$SGLANG_WORK"
# One Nsight session covers the whole process tree, including all DP workers.
nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none \
  -o "$report_dir/qwen30b-dpa4-ep4-b512-$graph_mode" \
  python -m sglang.launch_server "${server_args[@]}" \
  2>&1 | tee "$report_dir/server.log"
