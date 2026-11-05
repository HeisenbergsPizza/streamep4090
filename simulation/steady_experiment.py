"""One reproducible open-arrival capacity experiment, followed by full drain.

Arrival times and whole-request routing resampling are independent of policy.
Run cases with identical workload/cost parameters and explicit execution modes.
No statement in this module treats the proxy costs as calibrated GPU performance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

from .overheads import BatchOverheadModel, add_overhead_arguments, overhead_config_from_args


RUNTIME_REFERENCE_PATH = (Path(__file__).resolve().parents[1]
                          / "experiments/scripts/delta/eval/config.sh")
_RUNTIME_PARAMETERS = {
    "attention_batch_cap": ("MAX_BATCH_SIZE_ATTN", int),
    "expert_batch_cap": ("MAX_BATCH_SIZE_EXP", int),
    "lookahead_steps": ("DEFRAG_LOOKAHEAD_STEPS", int),
    "lookback_steps": ("DEFRAG_LOOKBACK_STEPS", int),
    "weight_decay": ("DEFRAG_WEIGHT_DECAY", float),
}


def load_runtime_reference():
    """Read literal runtime settings without executing the deployment script."""
    path = RUNTIME_REFERENCE_PATH.resolve()
    contents = path.read_bytes()
    source = contents.decode("utf-8")
    parameters = {}
    for name, (variable, convert) in _RUNTIME_PARAMETERS.items():
        assignments = re.findall(rf"^{variable}=([^\n#]+?)[ \t]*(?:#.*)?$", source, re.MULTILINE)
        if len(assignments) != 1:
            raise ValueError(f"runtime reference requires one literal {variable} assignment: {path}")
        try:
            parameters[name] = convert(assignments[0].strip())
        except ValueError as exc:
            raise ValueError(f"runtime reference {variable} must be a numeric literal: {path}") from exc
    if (parameters["attention_batch_cap"] < 1 or parameters["expert_batch_cap"] < 1
            or parameters["lookahead_steps"] < 1 or parameters["lookback_steps"] < 0
            or not 0 <= parameters["weight_decay"] <= 1):
        raise ValueError(f"invalid runtime reference parameters: {path}")
    return {
        "source": {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()},
        "parameters": parameters,
        "scope": "batch caps and defrag scoring parameters only",
        "complete_runtime_equivalence_verified": False,
        "not_verified_by_this_reference": [
            "GPU topology and DP/EP/TP placement", "model, dtype, routing and operator shapes",
            "KV memory admission and prefill", "request lengths and arrival workload",
            "attention/expert kernel and communication cost calibration",
            "CPU scheduling and GPU reformation overhead calibration; explicit CPU/GPU resource overlap",
        ],
    }


def poisson_workload(rate_rps, duration_ms, pool_size, seed):
    """Return positive Poisson arrivals before the horizon and profile ordinals.

    Independent RNG streams preserve the same request sequence across rates and
    policies. A sampled request keeps all of its token/layer/Top-k routing rows.
    """
    if not math.isfinite(rate_rps) or rate_rps <= 0:
        raise ValueError("rate_rps must be finite and positive")
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("duration_ms must be finite and positive")
    if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    arrivals_rng, routes_rng = random.Random(seed), random.Random(seed + 1000003)
    arrivals, indices = [], []
    now = 0.0
    while True:
        now += arrivals_rng.expovariate(rate_rps / 1000)
        if now >= duration_ms:
            break
        arrivals.append(now)
        indices.append(routes_rng.randrange(pool_size))
    return arrivals, indices


def measurement_windows(warmup_ms, end_ms, block_ms):
    if not 0 <= warmup_ms < end_ms or block_ms <= 0:
        raise ValueError("require 0 <= warmup < end and a positive block size")
    count = (end_ms - warmup_ms) / block_ms
    if not math.isclose(count, round(count), rel_tol=0, abs_tol=1e-9):
        raise ValueError("measurement duration must contain whole blocks")
    return [("measurement", warmup_ms, end_ms)] + [
        (f"block_{i:03d}", warmup_ms + i * block_ms, warmup_ms + (i + 1) * block_ms)
        for i in range(round(count))]


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def build_parser():
    defaults = load_runtime_reference()["parameters"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-profile", required=True, type=Path)
    parser.add_argument("--expert-profile", required=True, type=Path)
    parser.add_argument("--attention-profile", type=Path,
                        help="pure-decode batch/context CSV in milliseconds; optional component column")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("async", "sync", "tbo"), default="async",
                        help="async queue scheduling, synchronous EP, or two-microbatch overlap")
    parser.add_argument("--policy", choices=("defrag", "flfs", "mtfs", "fifo", "sync", "tbo"),
                        help="required for async; sync/tbo default to the matching mode")
    parser.add_argument("--expert-scheduling", choices=("layer", "expert"), default="layer",
                        help="expert is a simulation-only extension; layer matches runtime granularity")
    parser.add_argument("--rate", type=float, required=True, help="offered requests/second, positive")
    parser.add_argument("--duration-s", type=float, default=40)
    parser.add_argument("--warmup-s", type=float, default=10)
    parser.add_argument("--block-s", type=float, default=5)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--active-request-cap", type=int, default=128)
    parser.add_argument("--attention-batch-cap", type=int, default=defaults["attention_batch_cap"],
                        help="default follows the real runtime reference; explicit overrides are recorded")
    parser.add_argument("--expert-batch-cap", type=int, default=defaults["expert_batch_cap"],
                        help="default follows the real runtime reference; explicit overrides are recorded")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--initial-context", type=int, default=1024)
    parser.add_argument("--attention-per-token-us", type=float, default=20)
    parser.add_argument("--expert-cost-scale", type=float, default=1)
    parser.add_argument("--network-bandwidth-gbps", type=float, default=800)
    parser.add_argument("--network-latency-us", type=float, default=5)
    parser.add_argument("--ticks-per-ms", type=float, default=10)
    parser.add_argument("--lookahead-steps", type=int, default=defaults["lookahead_steps"])
    parser.add_argument("--lookback-steps", type=int, default=defaults["lookback_steps"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--sample-interval-ms", type=float, default=100)
    parser.add_argument("--trace-duration-ms", type=float, default=200,
                        help="detailed queue/decision window at start of final measurement block; 0 disables")
    parser.add_argument("--trace-interval-ms", type=float, default=.1)
    parser.add_argument("--token-trace", action="store_true",
                        help="async only: follow tokens selected in the detailed trace window through expert returns, Top-k join, and next Attention selection")
    add_overhead_arguments(parser)
    return parser


def resolve_execution(args):
    """Reject ambiguous policy/mode combinations before creating artifacts."""
    mode = args.mode
    if mode == "async":
        if args.policy not in ("defrag", "flfs", "mtfs", "fifo"):
            raise ValueError("--mode async requires --policy defrag, flfs, mtfs, or fifo")
    else:
        if args.expert_scheduling != "layer":
            raise ValueError("sync/tbo require --expert-scheduling layer")
        if args.policy not in (None, mode):
            raise ValueError(f"--mode {mode} requires --policy {mode}, or omit --policy")
        args.policy = mode
    return mode


def run_case(args):
    from .sim_async import run_simulation
    from .steady_metrics import SteadyMetricsObserver
    from .run import _network_delay, _scheduling_provenance
    import pyarrow.parquet as pq

    mode = resolve_execution(args)
    overhead_config = overhead_config_from_args(args)
    overhead_description = BatchOverheadModel(
        ticks_per_millisecond=args.ticks_per_ms, **overhead_config).describe()
    reference = load_runtime_reference()
    reference["matched_parameters"] = [
        name for name, value in reference["parameters"].items() if getattr(args, name) == value]
    reference["overrides"] = {
        name: {"reference": value, "actual": getattr(args, name)}
        for name, value in reference["parameters"].items() if getattr(args, name) != value}
    scheduling = _scheduling_provenance(args.expert_scheduling)
    if mode != "async":
        scheduling.update(
            execution_mode=mode, simulation_only_extension=True,
            baseline="synchronous_ep" if mode == "sync" else "two_microbatch_overlap_ep",
            microbatches=1 if mode == "sync" else 2,
            admission="iteration_boundary",
            compute="one global collective compute phase at a time; per-GPU serial local batches within the phase",
            communication="shared per-destination intra/inter receiver FIFO resources; may overlap compute",
            cost_models="same Attention and Expert profile functions as async",
            unused_defrag_parameters=["lookahead_steps", "lookback_steps", "weight_decay"],
            runtime_execution_equivalence_verified=False)
    reference["scheduling"] = scheduling

    # Validate all study-specific settings before creating an output directory.
    for key in ("rate", "duration_s", "block_s", "sample_interval_ms", "trace_interval_ms",
                "network_bandwidth_gbps", "ticks_per_ms", "expert_cost_scale"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("warmup_s", "trace_duration_ms", "attention_per_token_us", "network_latency_us"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("gpus", "layers", "experts", "top_k", "output_tokens", "active_request_cap",
                "attention_batch_cap", "expert_batch_cap", "hidden_dim", "bytes_per_element"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    end_ms, warmup_ms, block_ms = args.duration_s * 1000, args.warmup_s * 1000, args.block_s * 1000
    windows = measurement_windows(warmup_ms, end_ms, block_ms)
    trace_start = end_ms - block_ms
    if args.trace_duration_ms > block_ms:
        raise ValueError("trace-duration-ms must fit within the final measurement block")
    if getattr(args, "token_trace", False) and (mode != "async" or args.trace_duration_ms <= 0):
        raise ValueError("--token-trace requires --mode async and a positive --trace-duration-ms")
    if args.experts % args.gpus or args.top_k > args.experts:
        raise ValueError("experts must divide GPUs and cover top-k")
    profile_ids = sorted(set(pq.read_table(args.routing_profile, columns=["rid"])["rid"].to_pylist()))
    arrivals, mapping = poisson_workload(args.rate, end_ms, len(profile_ids), args.seed)
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output directory must be absent or empty: {output}")
    routing_identity, expert_identity = file_identity(args.routing_profile), file_identity(args.expert_profile)
    attention_identity = file_identity(args.attention_profile) if args.attention_profile else None
    output.mkdir(parents=True, exist_ok=True)
    workload_path = output / "workload.csv"
    with workload_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["request_id", "arrival_ms", "profile_request_ordinal", "profile_request_id"])
        writer.writerows((i, t, j, profile_ids[j]) for i, (t, j) in enumerate(zip(arrivals, mapping)))
    configuration = {key: str(value.resolve()) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    configuration.update(execution_mode=mode, rate_rps=args.rate, arrival_end_ms=end_ms, warmup_ms=warmup_ms,
                         block_ms=block_ms, total_requests=len(arrivals),
                         measurement_windows_ms=windows, routing_identity=routing_identity,
                         expert_identity=expert_identity, workload_identity=file_identity(workload_path),
                         attention_identity=attention_identity,
                         routing_pool_size=len(profile_ids), source_kind="bootstrap_existing_request_prefixes",
                         routing_replay="whole-request sampling with replacement; original token/layer/Top-k order; no token wrapping",
                         hardware_calibrated=False, grouped_profile=None,
                         finalize_per_token_us=0, n_gpu_per_host=args.gpus,
                         first_arrival_ms=arrivals[0] if arrivals else None,
                         last_arrival_ms=arrivals[-1] if arrivals else None)
    configuration["runtime_reference"] = reference
    configuration["scheduling"] = scheduling
    configuration["overhead_config"] = overhead_config
    configuration["cost_assumptions"] = {"batch_overhead": overhead_description}
    configuration["code_identity"] = {
        name: file_identity(Path(__file__).parent / name)["sha256"]
        for name in ("steady_experiment.py", "steady_metrics.py", "sim_async.py", "sim_collective.py", "router.py",
                     "costs.py", "overheads.py", "schedulers.py", "metrics.py", "queue_trace.py", "token_trace.py", "util.py", "run.py")}
    configuration["comparison_configuration"] = {
        key: configuration[key] for key in (
            "execution_mode", "gpus", "layers", "experts", "top_k", "output_tokens", "active_request_cap",
            "attention_batch_cap", "expert_batch_cap", "expert_scheduling", "hidden_dim", "bytes_per_element",
            "initial_context", "attention_per_token_us", "expert_cost_scale", "network_bandwidth_gbps",
            "network_latency_us", "ticks_per_ms", "lookahead_steps", "lookback_steps", "weight_decay",
            "routing_identity", "expert_identity", "routing_pool_size", "source_kind", "routing_replay",
            "hardware_calibrated", "attention_profile", "attention_identity", "grouped_profile", "finalize_per_token_us", "n_gpu_per_host",
            "overhead_config", "cost_assumptions", "code_identity")}
    configuration["comparison_configuration"]["runtime_reference_identity"] = reference["source"]
    write_json(output / "configuration.json", configuration)
    write_json(output / "status.json", {"status": "running", "complete": False})
    observer = SteadyMetricsObserver(measurement_windows_ms=windows,
                                    sample_interval_ms=args.sample_interval_ms,
                                    request_log_path=output / "requests.csv")
    previous = {}
    began = time.perf_counter()
    with (output / "tokens.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["request_id", "token_index", "token_id", "home_gpu", "birth_ms", "completion_ms",
                         "itl_ms", "attention_ready_wait_ms", "expert_branch_ready_wait_sum_ms",
                         "join_gap_sum_ms"])

        def begin(token):
            token.sampled_for_stats = True

        def complete(token, now):
            completion_ms = now / args.ticks_per_ms
            itl = completion_ms - previous[token.request_id] if token.token_index else None
            previous[token.request_id] = completion_ms
            writer.writerow([token.request_id, token.token_index, token.tid, token.home_gpu,
                             token.birth_time / args.ticks_per_ms, completion_ms, itl,
                             token.attention_ready_wait / args.ticks_per_ms,
                             token.expert_ready_wait / args.ticks_per_ms, token.join_wait / args.ticks_per_ms])
            if token.token_index == args.output_tokens - 1:
                del previous[token.request_id]

        trace_options = (dict(queue_trace_path=output / "scheduler_queue_trace",
                              queue_trace_interval_ms=args.trace_interval_ms,
                              queue_trace_start_ms=trace_start,
                              queue_trace_end_ms=trace_start + args.trace_duration_ms)
                         if args.trace_duration_ms else {})
        if getattr(args, "token_trace", False):
            trace_options.update(token_trace_path=output / "token_progress_trace",
                                 token_trace_start_ms=trace_start,
                                 token_trace_end_ms=trace_start + args.trace_duration_ms)
        result = run_simulation(
            ep_group_size=args.gpus, attn_dp_group_size=args.gpus, n_gpu_per_host=args.gpus,
            total_expert_count=args.experts, num_layers=args.layers, routing_top_k=args.top_k,
            total_requests=len(arrivals), tokens_per_request=args.output_tokens,
            arrival_rate=0, arrival_times_ms=arrivals, routing_request_indices=mapping,
            observer=observer, global_request_max_batch_size=args.active_request_cap,
            max_attention_batch_size=args.attention_batch_cap, max_batch_size=args.expert_batch_cap,
            expert_profile_path=str(args.expert_profile), profile_routing_path=str(args.routing_profile),
            attention_profile_path=str(args.attention_profile) if args.attention_profile else None,
            initial_context_length=args.initial_context, expert_cost_scale=args.expert_cost_scale,
            attn_service_t=args.attention_per_token_us / 1000 * args.ticks_per_ms,
            finalize_per_token_t=0, net_delay_fn=_network_delay(args), ticks_per_millisecond=args.ticks_per_ms,
            overhead_config=overhead_config,
            hidden_dim=args.hidden_dim, bytes_per_element=args.bytes_per_element,
            execution_mode=mode,
            rng_seed=args.seed, scheduling_policy=args.policy, expert_scheduling=args.expert_scheduling,
            defrag_lookahead_steps=args.lookahead_steps,
            defrag_lookback_steps=args.lookback_steps, defrag_weight_decay=args.weight_decay,
            per_token_begin_cb=begin, per_token_stats_cb=complete, schedule_trace_limit=64,
            metric_sample_capacity=100000, quiet=True, **trace_options)
    elapsed = time.perf_counter() - began
    residuals = {key: result[key] for key in ("active_requests_at_end", "waiting_requests_at_end",
                 "unarrived_requests_at_end", "inflight_tokens_at_end", "incomplete_joins_at_end",
                 "inflight_network_branches_at_end", "ready_attention_at_end", "ready_expert_at_end")}
    drained = (not any(residuals.values()) and not result["stopped_early"]
               and result["completed_tokens"] == len(arrivals) * args.output_tokens
               and result["completed_requests"] == len(arrivals))
    envelope = {"format": "streaminfer.steady_experiment", "version": 1,
                "configuration": configuration, "simulation": result,
                "validation": {"fully_drained": drained, "residuals": residuals},
                "drain_ms": max(0., result["makespan_ms"] - end_ms), "wall_time_seconds": elapsed}
    write_json(output / "result.json", envelope)
    write_json(output / "status.json", {"status": "complete" if drained else "failed", "complete": drained})
    if not drained:
        raise RuntimeError("finite workload failed to drain; result saved for inspection")
    print(f"{args.policy} rate={args.rate:g} seed={args.seed}: {len(arrivals)} requests, "
          f"drain={envelope['drain_ms'] / 1000:.3f}s, wall={elapsed:.1f}s, {output}", flush=True)
    return envelope


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_case(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
