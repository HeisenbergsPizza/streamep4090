#!/usr/bin/env python3
"""Own one local Ray cluster, then invoke the repository's benchmark/server."""
import argparse
import asyncio
from datetime import datetime
import errno
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import sys
import tempfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]


def parse_configuration(argv=None):
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("mode", choices=("smoke", "benchmark", "server"))
    wrapper.add_argument("--dry-run", action="store_true")
    options, extra = wrapper.parse_known_args(argv)
    # Read the real parser without importing torch, CUDA, Ray, or vLLM.
    spec = importlib.util.spec_from_file_location("rtx4090_benchmark_cli", REPO / "benchmark/utils.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    report = f"reports/rtx4090/{options.mode}-{datetime.now():%Y%m%d-%H%M%S}.csv"
    defaults = [
        "--model", "qwen3_30b", "--num-nodes", "1", "--num-gpus", "4",
        "--dp-size", "4", "--ep-size", "4", "--tp-size", "1", "--placement", "colocate",
        "--num-layers", "12" if options.mode == "smoke" else "48",
        "--num-experts", "128", "--topk", "8", "--num-kv-heads", "4",
        "--less-than-sm90", "--attn-qkv-quant", "none", "--moe-linear-quant", "none",
        "--max-batch-size-attn", "16", "--max-attn-graph-bsz", "16", "--max-batch-size-expert", "128",
        "--gpu-usage", "0.75", "--block-size", "16", "--max-pending-sends", "16",
        "--transport", "zmq", "--unified-scheduler-type", "flfs", "--dp-policy", "max",
        "--generator-type", "uniform", "--rate", "1", "--num-requests", "8" if options.mode == "smoke" else "32",
        "--min-input-len", "64", "--max-input-len", "65",
        "--min-output-len", "8" if options.mode == "smoke" else "32",
        "--max-output-len", "9" if options.mode == "smoke" else "33", "--file", report,
    ]
    cli_args = defaults + extra
    args = cli.get_parser_base().parse_args(cli_args)
    for field, expected in {"model": "qwen3_30b", "num_nodes": 1, "num_gpus": 4, "dp_size": 4, "ep_size": 4, "tp_size": 1, "placement": "colocate", "transport": "zmq"}.items():
        if getattr(args, field) != expected:
            wrapper.error(f"This launcher requires {field}={expected}")
    if args.num_experts != 128 or args.topk != 8 or args.num_kv_heads != 4:
        wrapper.error("Keep qwen3_30b experts=128, topk=8 and KV heads=4")
    if args.attn_qkv_quant != "none" or args.moe_linear_quant != "none":
        wrapper.error("This deployment profile uses BF16; quantization needs separate validation")
    if args.num_requests < 1 or args.rate <= 0:
        wrapper.error("num-requests and rate must be positive")
    if not (0 < args.gpu_usage < 1):
        wrapper.error("gpu-usage must be between 0 and 1")
    if not (1 <= args.num_layers <= 48):
        wrapper.error("num-layers must be between 1 and 48")
    if args.min_input_len < 1 or args.max_input_len <= args.min_input_len or args.min_output_len < 1 or args.max_output_len <= args.min_output_len:
        wrapper.error("Lengths use [min,max); positive minimum lengths are required")
    if args.max_input_len + args.max_output_len - 2 > 4096:
        wrapper.error("Maximum context length (input + output) must not exceed 4096")
    if args.max_batch_size_attn < 1 or args.max_batch_size_expert < 1 or args.max_pending_sends < 1:
        wrapper.error("Batch limits and pending send count must be positive")
    if args.cuda_graph_attn and args.max_attn_graph_bsz < args.max_batch_size_attn:
        wrapper.error("max-attn-graph-bsz must cover max-batch-size-attn")
    if args.nsys and args.profile_dir is not None:
        wrapper.error("Use either --nsys or --profile-dir, not both CUPTI collectors")
    if args.nsys and options.mode == "server":
        wrapper.error("Nsight capture supports one smoke/benchmark run per launch; use benchmark mode")
    return options, args, cli_args


def check_tcp_port(port, *, reuse_address=False):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if reuse_address:
                # Match libzmq's TCP listener: permit reusable closed connections,
                # but never enable SO_REUSEPORT or share a live listener.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            sock.listen(1)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise RuntimeError(
                f"Required local port {port} is unavailable ({exc}). "
                f"Inspect: ss -ltnp 'sport = :{port}' and ss -tanop 'sport = :{port}'. "
                "Identify the owner/state before stopping a confirmed prior run."
            ) from exc
        raise RuntimeError(f"Cannot probe required local port {port}: {exc}") from exc


def check_ports(server):
    # All these endpoints are ZeroMQ TCP listeners with SO_REUSEADDR enabled.
    for port in [22000, 22001, 22002, 22003, 23000, 24000, 24016, 24032, 24048]:
        check_tcp_port(port, reuse_address=True)
    check_tcp_port(int(os.environ.get("MASTER_PORT", "26500")))
    if server:
        check_tcp_port(6699)


def main(argv=None):
    options, args, cli_args = parse_configuration(argv)
    print(json.dumps({"mode": options.mode, "configuration": vars(args)}, indent=2), flush=True)
    if options.dry_run:
        print("CLI validation passed; no GPU, Ray cluster, download, or file write was performed.")
        return
    if args.nsys:
        if shutil.which("nsys") is None:
            raise RuntimeError("Install Nsight Systems CLI and put nsys on PATH; see docs/rtx4090-profiling.md")
        # Set before importing controller/env/native extension in this process.
        os.environ["DMOE_ENABLE_NVTX"] = "1"
        os.environ["ENABLE_NVTX"] = "0"
    os.chdir(REPO)
    assert sys.platform == "linux" and sys.version_info[:2] == (3, 12), "Use Linux + Python 3.12"
    import fcntl
    (REPO / "reports/rtx4090").mkdir(parents=True, exist_ok=True)
    lock = open(REPO / "reports/rtx4090/launcher.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another RTX4090 launcher is running from this checkout") from exc
    check_ports(options.mode == "server")
    import torch
    import disagmoe_c as c
    if args.nsys and (not hasattr(c, "nvtx_enabled") or not c.nvtx_enabled()):
        raise RuntimeError("Rebuild the updated extension for real NVTX ranges; see docs/rtx4090-profiling.md")
    assert torch.cuda.device_count() == 4
    assert not c.gdrcopy_compiled() and not c.gdrcopy_enabled(), "Rebuild/run the no-GDR configuration"
    assert os.environ.get("NCCL_P2P_DISABLE") == "1" and os.environ.get("NCCL_IB_DISABLE") == "1", "Use launch.sh to set the measured-topology environment"
    assert os.environ.get("NCCL_SHM_DISABLE") != "1"
    cpus = len(os.sched_getaffinity(0))
    assert cpus >= 16, "Need >=16 available logical CPUs for the Ray actors"
    import ray
    assert not ray.is_initialized(), "Launcher must own its Ray instance"
    # A user-provided RAY_ADDRESS must never redirect this single-node job.
    os.environ.pop("RAY_ADDRESS", None)
    ray_dir = tempfile.mkdtemp(prefix="streamep-ray-", dir="/tmp")
    os.environ["RAY_TMPDIR"] = ray_dir
    print(f"Creating an independent local Ray cluster; logs: {ray_dir}", flush=True)

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    Path(args.file).parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "benchmark"))
    sys.path.insert(0, str(REPO))
    try:
        ray.init(address="local", num_cpus=cpus, num_gpus=4, include_dashboard=False,
                 _temp_dir=ray_dir, object_store_memory=512 * 2**20)
        alive_nodes = [node for node in ray.nodes() if node["Alive"]]
        assert len(alive_nodes) == 1 and ray.cluster_resources().get("GPU") == 4
        if options.mode == "server":
            import server
            sys.argv = [str(REPO / "benchmark/server.py")] + cli_args
            server.main()
        else:
            # Avoid benchmark_serving.main(), which catches errors and exits 0.
            import benchmark_serving as benchmark
            master = benchmark.launch(args)
            metrics = asyncio.run(benchmark.benchmark_serving(master, args))
            assert metrics.req_throughput > 0 and metrics.e2e_duration > 0, "Invalid benchmark result"
            assert Path(args.file).is_file(), "Benchmark did not write its result file"
            print(metrics, flush=True)
            print(f"PASS: {options.mode} completed; metrics: {args.file}", flush=True)
            if args.nsys:
                from disagmoe.utils.nsys_profile import collect_nsight_reports, make_nsight_config
                profile_dir = Path(os.environ.get(
                    "DMOE_PROFILE_OUTPUT_DIR", str(Path(args.file).with_suffix(".nsys")))).resolve()
                reports = collect_nsight_reports(ray_dir, profile_dir)
                manifest = {
                    "workload": "synthetic decode with random weights and routing",
                    "configuration": vars(args), "ray_directory": ray_dir,
                    "nsys_options": make_nsight_config(
                        cpu_sampling=os.environ.get("DMOE_NSYS_CPU_SAMPLING", "0") == "1"),
                    "reports": [str(path) for path in reports],
                    "workers": getattr(master, "nsys_workers", []),
                    "coverage": "GPU Engine actors; driver/tokenizer/detokenizer CPU not captured",
                    "capture_window": "after warmup until all benchmark responses arrive",
                }
                (profile_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"PROFILE: copied {len(reports)} reports to {profile_dir}; inspect with nsys stats/GUI.", flush=True)
    finally:
        ray.shutdown()  # Only this process's locally created cluster.
        lock.close()


if __name__ == "__main__":
    main()
