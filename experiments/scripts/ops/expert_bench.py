#!/usr/bin/env python3
"""Single-GPU BF16 expert-kernel comparison; no Ray, server, or NCCL.

prepare writes ONE CPU tensor fixture for both environments. run validates the
native forward against an FP32 reference before timing or collecting counters.
Only one post-warmup invocation carries ExpertBench.* NVTX markers.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

SCHEMA_VERSION = 1
STAGES = ("up", "down", "activation", "full")


def validate_config(config):
    for name in ("tokens", "hidden", "intermediate", "global_experts", "ep_size", "top_k"):
        value = config[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["global_experts"] % config["ep_size"]:
        raise ValueError("global_experts must be divisible by ep_size")
    if type(config["local_rank"]) is not int or not 0 <= config["local_rank"] < config["ep_size"]:
        raise ValueError("local_rank must be in [0, ep_size)")
    if config["top_k"] > config["global_experts"]:
        raise ValueError("top_k exceeds global_experts")
    if config["hidden"] % 8 or config["intermediate"] % 8:
        raise ValueError("hidden and intermediate must be multiples of 8 for CUTLASS alignment")
    if config["routing"] not in ("balanced", "random"):
        raise ValueError("routing must be balanced or random")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")


def make_routes(config):
    """Global expert IDs, unique within each token; independent of torch RNG."""
    validate_config(config)
    rng = random.Random(config["seed"])
    experts = list(range(config["global_experts"]))
    if config["routing"] == "random":
        return [rng.sample(experts, config["top_k"]) for _ in range(config["tokens"])]
    rng.shuffle(experts)
    return [[experts[(t * config["top_k"] + k) % len(experts)]
             for k in range(config["top_k"])] for t in range(config["tokens"])]


def localize_routes(routes, global_experts, ep_size, local_rank):
    if (global_experts < 1 or ep_size < 1 or global_experts % ep_size
            or not 0 <= local_rank < ep_size):
        raise ValueError("Invalid global_experts/ep_size/local_rank")
    local_experts = global_experts // ep_size
    first = local_rank * local_experts
    counts = [0] * local_experts
    mapped = []
    for row in routes:
        if len(set(row)) != len(row) or any(type(e) is not int or not 0 <= e < global_experts for e in row):
            raise ValueError("Each token must have unique, in-range global expert IDs")
        local_row = []
        for expert in row:
            value = expert - first if first <= expert < first + local_experts else -1
            local_row.append(value)
            if value >= 0:
                counts[value] += 1
        mapped.append(local_row)
    return mapped, counts


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def prepare(args):
    import torch
    config = {key: getattr(args, key) for key in (
        "tokens", "hidden", "intermediate", "global_experts", "ep_size", "local_rank",
        "top_k", "seed", "routing")}
    validate_config(config)
    output = Path(args.output).expanduser().resolve()
    manifest = output.with_suffix(output.suffix + ".json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Choose a new case path; refusing to replace {output} or {manifest}")
    global_ids = make_routes(config)
    local_ids, counts = localize_routes(global_ids, config["global_experts"], config["ep_size"], config["local_rank"])
    if not sum(counts):
        raise ValueError("This fixture has no local expert branches; choose more tokens or another rank")
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    def normal(*shape, scale=1.0):
        return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(torch.bfloat16)
    h, inter, experts = config["hidden"], config["intermediate"], len(counts)
    case = {
        "schema_version": SCHEMA_VERSION, "config": config,
        "x": normal(config["tokens"], h),
        "w13": normal(experts, h, 2 * inter, scale=1 / math.sqrt(h)),
        "w2": normal(experts, inter, h, scale=1 / math.sqrt(inter)),
        "global_topk_ids": torch.tensor(global_ids, dtype=torch.int64),
        "topk_ids": torch.tensor(local_ids, dtype=torch.int64),
        "topk_weights": torch.full((config["tokens"], config["top_k"]), 1 / config["top_k"], dtype=torch.float32),
        "batch_sizes": torch.tensor(counts, dtype=torch.int64),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(case, handle)
    summary = {"schema_version": SCHEMA_VERSION, "config": config, "case_sha256": digest(output),
               "local_branches": sum(counts), "per_expert_rows": counts,
               "scope": "One synthetic BF16 layer, one EP rank; no checkpoint or communication"}
    write_json(manifest, summary)
    print(json.dumps(summary, indent=2))


def load_case(path):
    import torch
    case = torch.load(path, map_location="cpu", weights_only=True)
    if case.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported expert fixture schema")
    config = case["config"]
    validate_config(config)
    t, h, inter, k = (config[key] for key in ("tokens", "hidden", "intermediate", "top_k"))
    e = config["global_experts"] // config["ep_size"]
    expected = {"x": ((t, h), torch.bfloat16), "w13": ((e, h, 2 * inter), torch.bfloat16),
                "w2": ((e, inter, h), torch.bfloat16), "topk_ids": ((t, k), torch.int64),
                "global_topk_ids": ((t, k), torch.int64), "topk_weights": ((t, k), torch.float32),
                "batch_sizes": ((e,), torch.int64)}
    for name, (shape, dtype) in expected.items():
        tensor = case[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"Invalid fixture tensor {name}: expected {shape}, {dtype}")
        if not tensor.is_contiguous() or not torch.isfinite(tensor).all().item():
            raise ValueError(f"Fixture tensor {name} must be contiguous and finite")
    local, counts = localize_routes(case["global_topk_ids"].tolist(), config["global_experts"], config["ep_size"], config["local_rank"])
    if local != case["topk_ids"].tolist() or counts != case["batch_sizes"].tolist() or not sum(counts):
        raise ValueError("Fixture local routes/counts do not match global routes")
    if (case["topk_weights"] < 0).any().item() or not torch.allclose(case["topk_weights"].sum(1), torch.ones(t)):
        raise ValueError("Top-k weights must be nonnegative and sum to one")
    return case


def build_streaminfer_runner(case, stage):
    """Use the actual extension kernels; mirror MoEExpertsCUTLASS BF16 forward.

    Packing and validation-only combination are outside this native core.
    Avoid importing unrelated vLLM/Ray/serving dependencies into this process.
    """
    import contextlib
    import torch
    import disagmoe_c
    config = case["config"]
    description = disagmoe_c.init_grouped_gemm(torch.cuda.current_device())
    token_ids, slots = torch.where(case["topk_ids"] >= 0)
    expert_ids = case["topk_ids"][token_ids, slots]
    order = torch.argsort(expert_ids, stable=True)
    token_ids, slots = token_ids[order], slots[order]
    hiddens = case["x"][token_ids].contiguous()
    branches = hiddens.size(0)
    up_runner = disagmoe_c.CutlassGemmRunner(case["w13"], branches)
    down_runner = disagmoe_c.CutlassGemmRunner(case["w2"], branches)
    act = torch.nn.SiLU(inplace=True)
    inter, h = config["intermediate"], config["hidden"]
    def scope(label, profile):
        return torch.cuda.nvtx.range(f"ExpertBench.{label}") if profile and stage == label else contextlib.nullcontext()
    def callback(*, profile=False):
        with scope("full", profile):
            up_out = torch.empty((branches, 2 * inter), dtype=torch.bfloat16, device=hiddens.device)
            down_out = torch.empty((branches, h), dtype=torch.bfloat16, device=hiddens.device)
            with scope("up", profile):
                up_runner.setup_meta(hiddens, up_out, case["batch_sizes"])
                up_runner.run()
            with scope("activation", profile):
                up = act(up_out[:, :inter]) * up_out[:, inter:]
            with scope("down", profile):
                down_runner.setup_meta(up, down_out, case["batch_sizes"])
                down_runner.run()
        return down_out
    def normalize_output(branch_output):
        # Validation only: restore token/top-k slots, weight, and sum locally.
        # This is deliberately excluded from the native kernel timing/capture.
        slots_out = torch.zeros((config["tokens"], config["top_k"], h), device=hiddens.device, dtype=torch.float32)
        slots_out[token_ids, slots] = branch_output.float() * case["topk_weights"][token_ids, slots, None]
        return slots_out.sum(dim=1)
    callback.normalize_output = normalize_output
    metadata = {"implementation": "disagmoe_c.CutlassGemmRunner + original torch SiLU/multiply sequence",
                "cutlass_config": description, "local_branches": branches,
                "timing_scope": "Native expert core only: 2 metadata kernels, 2 GEMMs, SiLU, multiply. Input packing and output weighting/combination excluded.",
                "weight_layout": "[E,K,N]", "cuda_graph": False}
    return callback, metadata


def reference(case):
    import torch
    output = torch.zeros_like(case["x"], dtype=torch.float32)
    h = case["config"]["intermediate"]
    # FP32 reference with TF32 disabled. This executes outside NCU capture.
    for expert in range(case["w13"].shape[0]):
        rows, slots = torch.where(case["topk_ids"] == expert)
        if rows.numel() == 0:
            continue
        up = case["x"][rows].float() @ case["w13"][expert].float()
        gated = torch.nn.functional.silu(up[:, :h]) * up[:, h:]
        down = gated @ case["w2"][expert].float()
        # Each token has at most one branch for this expert: no ambiguous writes.
        output[rows] += down * case["topk_weights"][rows, slots, None]
    return output


def compare_output(actual, expected):
    import torch
    actual, expected = actual.double(), expected.double()
    if (actual.shape != expected.shape or not torch.isfinite(actual).all().item()
            or not torch.isfinite(expected).all().item()):
        raise AssertionError("Expert output/reference has wrong shape or nonfinite values")
    delta = actual - expected
    relative_l2 = (delta.norm() / expected.norm().clamp_min(1e-12)).item()
    relative_max = (delta.abs().max() / expected.abs().max().clamp_min(1e-12)).item()
    stats = {"relative_l2": relative_l2, "max_abs_error": delta.abs().max().item(),
             "max_error_over_reference_max": relative_max,
             "relative_l2_limit": 0.02, "max_error_over_reference_max_limit": 0.05}
    # Mixed BF16 rounding (fused vs separate activation/weighting) is expected.
    # Do not silently loosen gates or proceed to profile incorrect kernels.
    if not math.isfinite(relative_l2) or not math.isfinite(relative_max) or relative_l2 > 0.02 or relative_max > 0.05:
        raise AssertionError(f"BF16 expert validation failed: {stats}")
    return stats


def run(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace result: {output}")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; use prepare or --help on CPU")
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES controls the physical GPU.
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("This comparison targets SM89 (RTX4090); use the same SM89 GPU for both backends")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    cpu_case = load_case(args.case)
    case_hash = digest(args.case)
    case = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in cpu_case.items()}
    del cpu_case
    if args.backend == "streaminfer":
        # Direct script execution must find the repository's extension module.
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        callback, backend_metadata = build_streaminfer_runner(case, args.stage)
    else:
        from sglang_expert_adapter import build_sglang_runner
        callback, backend_metadata = build_sglang_runner(case, args.stage)
    normalize = getattr(callback, "normalize_output", lambda value: value)
    expected = reference(case)
    actual = normalize(callback())
    validation = compare_output(actual, expected)
    del actual, expected
    for _ in range(args.warmup):
        callback()
    torch.cuda.synchronize()
    timings = None
    if args.profile_only:
        # NCU --profile-from-start off excludes setup, reference, JIT, warmup.
        torch.cuda.profiler.start()
        try:
            captured_output = callback(profile=True)
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
        del captured_output
    else:
        values = []
        for _ in range(args.iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            timed_output = callback()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
            del timed_output
        timings = {"median": statistics.median(values), "min": min(values), "max": max(values), "samples": values}
    packages = {}
    for name in ("torch", "triton", "sglang", "sgl-kernel", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {"backend": args.backend, "stage": args.stage, "case": str(Path(args.case).resolve()),
              "case_sha256": case_hash, "config": case["config"],
              "per_expert_rows": case["batch_sizes"].cpu().tolist(),
              "local_branches": int(case["batch_sizes"].sum().item()),
              "device": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "cuda_graph": False,
              "cuda_runtime": torch.version.cuda, "packages": packages,
              "profile_only": args.profile_only, "nvtx_range": f"ExpertBench.{args.stage}",
              "warmup": args.warmup, "iterations": args.iterations,
              "validation": validation, "native_forward_ms": timings, "backend_metadata": backend_metadata,
              "note": "Native full-call boundaries differ. Compare matched GEMM counters/durations; these totals are not framework latency or equal-scope MoE timings."}
    write_json(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare", help="Save common synthetic CPU BF16 tensors once")
    make.add_argument("--output", required=True)
    for flag, default in (("tokens", 256), ("hidden", 2048), ("intermediate", 768), ("global-experts", 128),
                          ("ep-size", 4), ("local-rank", 0), ("top-k", 8), ("seed", 0)):
        make.add_argument(f"--{flag}", type=int, default=default)
    make.add_argument("--routing", choices=("balanced", "random"), default="balanced")
    measure = commands.add_parser("run", help="Validate, warm up, then time OR capture one native forward")
    measure.add_argument("--backend", choices=("streaminfer", "sglang"), required=True)
    measure.add_argument("--stage", choices=STAGES, default="up")
    measure.add_argument("--case", required=True)
    measure.add_argument("--output", required=True)
    measure.add_argument("--warmup", type=int, default=10)
    measure.add_argument("--iterations", type=int, default=20)
    measure.add_argument("--profile-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        (prepare if args.command == "prepare" else run)(args)
    except (ValueError, FileExistsError, RuntimeError, AssertionError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
