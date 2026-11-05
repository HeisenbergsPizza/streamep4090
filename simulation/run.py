"""CPU simulation CLI with explicit input provenance and comparable outputs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import random

from .metrics import Samples
from .overheads import BatchOverheadModel, add_overhead_arguments, overhead_config_from_args


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def _nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def _positive_float(value):
    value = _nonnegative_float(value)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("async", "sync", "tbo"), default="async")
    parser.add_argument("--policies", help="async only: comma-separated defrag,flfs,mtfs,fifo")
    parser.add_argument("--expert-scheduling", choices=("layer", "expert"), default="layer",
                        help="async expert candidate granularity; expert is a simulation-only extension")
    parser.add_argument("--synthetic", action="store_true", help="generate explicitly synthetic routing/cost fixtures")
    parser.add_argument("--routing-profile", type=Path)
    parser.add_argument("--expert-profile", type=Path)
    parser.add_argument("--attention-profile", type=Path,
                        help="pure-decode batch/context CSV in milliseconds; optional component column")
    parser.add_argument("--grouped-profile", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpus", type=_positive_int, default=2, help="logical GPUs on one host")
    parser.add_argument("--layers", type=_positive_int, default=4)
    parser.add_argument("--experts", type=_positive_int, default=8)
    parser.add_argument("--top-k", type=_positive_int, default=2)
    parser.add_argument("--requests", type=_positive_int, default=32)
    parser.add_argument("--output-tokens", type=_positive_int, default=8)
    parser.add_argument("--active-request-cap", type=_positive_int, default=16)
    parser.add_argument("--attention-batch-cap", type=_positive_int,
                        help="required with supplied profiles; synthetic smoke default: 16")
    parser.add_argument("--expert-batch-cap", type=_positive_int,
                        help="required with supplied profiles; synthetic smoke default: 64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=_nonnegative_float, default=0,
                        help="requests/second; 0 releases all requests at time zero")
    parser.add_argument("--initial-context", type=_nonnegative_int, default=0)
    parser.add_argument("--attention-per-token-us", type=_nonnegative_float, default=20,
                        help="unmeasured linear fallback when no attention profile is supplied")
    parser.add_argument("--finalize-per-token-us", type=_nonnegative_float, default=0,
                        help="unmeasured final A_L service cost; default excludes that cost")
    parser.add_argument("--expert-cost-scale", type=_positive_float, default=1.0)
    parser.add_argument("--lookahead-steps", type=_positive_int, default=4)
    parser.add_argument("--lookback-steps", type=_nonnegative_int, default=4)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=0.8)
    parser.add_argument("--network-bandwidth-gbps", type=_positive_float, default=100,
                        help="assumed per-receiver bandwidth in gigabits/second")
    parser.add_argument("--network-latency-us", type=_nonnegative_float, default=5,
                        help="assumed per-transfer startup latency; same-GPU transfer is zero")
    parser.add_argument("--hidden-dim", type=_positive_int, default=2880)
    parser.add_argument("--bytes-per-element", type=_positive_int, default=2)
    parser.add_argument("--ticks-per-ms", type=_positive_float, default=10)
    parser.add_argument("--tokens-after-saturation", type=_nonnegative_int)
    parser.add_argument("--schedule-trace-limit", type=_nonnegative_int, default=0,
                        help="maximum async decision records (TBO: timeline operations)")
    parser.add_argument("--queue-trace", action="store_true",
                        help="async only: record time-weighted ready queues and streaming decisions")
    parser.add_argument("--queue-trace-interval-ms", type=_positive_float, default=0.1,
                        help="time bin width for ready queue averages (milliseconds)")
    parser.add_argument("--queue-trace-start-ms", type=_nonnegative_float, default=0.0)
    parser.add_argument("--queue-trace-end-ms", type=_positive_float,
                        help="optional observation end; simulation continues to completion")
    parser.add_argument("--metric-sample-capacity", type=_positive_int, default=100000)
    parser.add_argument("--plot-itl-cdf", action="store_true")
    add_overhead_arguments(parser)
    return parser


def _synthetic_profiles(args, output_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq

    fixture_dir = output_dir / "synthetic-inputs"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for rid in range(args.requests):
        for token in range(args.output_tokens):
            for layer in range(args.layers):
                row = {"rid": rid, "token_index": token, "layer": layer}
                row.update({f"expert_logical_k{k}": expert for k, expert in
                            enumerate(rng.sample(range(args.experts), args.top_k))})
                rows.append(row)
    routing = fixture_dir / "synthetic-routing.parquet"
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b"provenance": b"synthetic; not a model trace"})
    pq.write_table(table, routing)
    expert = fixture_dir / "synthetic-expert.csv"
    with expert.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["batch_size", "avg_time_ms", "provenance"])
        for batch_size in range(1, args.expert_batch_cap + 1):
            writer.writerow([batch_size, 0.01 + 0.002 * batch_size, "synthetic_formula_not_measurement"])
    return routing, expert


def _file_record(path):
    if path is None:
        return None
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _network_delay(args):
    bytes_per_second = args.network_bandwidth_gbps * 1e9 / 8

    def delay(src, dst, num_tokens):
        if src == dst or num_tokens == 0:
            return 0.0
        seconds = args.network_latency_us * 1e-6 + (
            num_tokens * args.hidden_dim * args.bytes_per_element / bytes_per_second
        )
        return seconds * 1000 * args.ticks_per_ms

    return delay


def _scheduling_provenance(expert_scheduling):
    """Distinguish the runtime's layer groups from the explicit research extension."""
    if expert_scheduling not in ("layer", "expert"):
        raise ValueError("expert_scheduling must be layer or expert")
    return {
        "expert_scheduling": expert_scheduling,
        "runtime_expert_scheduling": "layer",
        "simulation_only_extension": expert_scheduling == "expert",
        "expert_batch_cap_scope": (
            "selected_gpu_layer_single_expert_branches" if expert_scheduling == "expert"
            else "selected_gpu_layer_expert_group_total_branches"),
    }


def _run_one(args, policy, routing, expert, output_dir, provenance):
    module = importlib.import_module(f".sim_{args.mode}", __package__)
    label = f"async_{policy}" if args.mode == "async" else args.mode
    token_path = output_dir / f"{label}_tokens.csv"
    itl = Samples(args.metric_sample_capacity, args.seed)
    latency = Samples(args.metric_sample_capacity, args.seed + 1)
    previous_completion = {}
    completed_requests = 0
    fields = ["mode", "policy", "synthetic", "request_id", "token_index", "token_id", "home_gpu",
              "birth_ticks", "completion_ticks", "token_execution_latency_ms", "itl_ms",
              "attention_ready_wait_ms", "finalize_ready_wait_ms",
              "expert_branch_ready_wait_sum_ms", "join_gap_sum_ms"]
    with token_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        def begin(token):
            token.sampled_for_stats = True

        def complete(token, completion):
            nonlocal completed_requests
            duration = (completion - token.birth_time) / args.ticks_per_ms
            latency.add(duration)
            interval = None
            if token.token_index > 0:
                interval = (completion - previous_completion[token.request_id]) / args.ticks_per_ms
                itl.add(interval)
            previous_completion[token.request_id] = completion
            if token.token_index == args.output_tokens - 1:
                completed_requests += 1
                del previous_completion[token.request_id]
            row = dict(mode=args.mode, policy=policy, synthetic=args.synthetic,
                       request_id=token.request_id, token_index=token.token_index, token_id=token.tid,
                       home_gpu=token.home_gpu, birth_ticks=token.birth_time, completion_ticks=completion,
                       token_execution_latency_ms=duration, itl_ms=interval)
            for column, attribute in (("attention_ready_wait_ms", "attention_ready_wait"),
                                      ("finalize_ready_wait_ms", "finalize_ready_wait"),
                                      ("expert_branch_ready_wait_sum_ms", "expert_ready_wait"),
                                      ("join_gap_sum_ms", "join_wait")):
                value = getattr(token, attribute, None)
                row[column] = value / args.ticks_per_ms if value is not None else None
            writer.writerow(row)

        common = dict(
            ep_group_size=args.gpus, attn_dp_group_size=args.gpus, n_gpu_per_host=args.gpus,
            total_expert_count=args.experts, num_layers=args.layers, routing_top_k=args.top_k,
            total_requests=args.requests, tokens_per_request=args.output_tokens,
            global_request_max_batch_size=args.active_request_cap,
            max_attention_batch_size=args.attention_batch_cap, max_batch_size=args.expert_batch_cap,
            arrival_rate=args.arrival_rate / (1000 * args.ticks_per_ms), rng_seed=args.seed,
            ticks_per_millisecond=args.ticks_per_ms,
            overhead_config=overhead_config_from_args(args),
            expert_profile_path=str(expert), profile_routing_path=str(routing),
            attention_profile_path=str(args.attention_profile) if args.attention_profile else None,
            grouped_profile_path=str(args.grouped_profile) if args.grouped_profile else None,
            expert_cost_scale=args.expert_cost_scale, initial_context_length=args.initial_context,
            attn_service_t=args.attention_per_token_us / 1000 * args.ticks_per_ms,
            finalize_per_token_t=args.finalize_per_token_us / 1000 * args.ticks_per_ms,
            net_delay_fn=_network_delay(args),
            tokens_after_reaching_max_bs=args.tokens_after_saturation,
            per_token_begin_cb=begin, per_token_stats_cb=complete,
        )
        if args.mode == "async":
            common.update(hidden_dim=args.hidden_dim, bytes_per_element=args.bytes_per_element,
                          scheduling_policy=policy, expert_scheduling=args.expert_scheduling,
                          defrag_lookahead_steps=args.lookahead_steps,
                          defrag_lookback_steps=args.lookback_steps, defrag_weight_decay=args.weight_decay,
                          metric_sample_capacity=args.metric_sample_capacity,
                          schedule_trace_limit=args.schedule_trace_limit, quiet=True)
            if args.queue_trace:
                common.update(queue_trace_path=output_dir / f"{label}_queue_trace",
                              queue_trace_interval_ms=args.queue_trace_interval_ms,
                              queue_trace_start_ms=args.queue_trace_start_ms,
                              queue_trace_end_ms=args.queue_trace_end_ms)
        elif args.mode == "tbo":
            common.update(timeline_capture_ops=args.schedule_trace_limit,
                          timeline_out_dir=str(output_dir / "tbo-timeline"), timeline_basename="tbo")
        result = module.run_simulation(**common)

    seconds = result["makespan"] / args.ticks_per_ms / 1000
    itl_summary = itl.summary()
    token_summary = latency.summary()
    summary = {
        "mode": args.mode, "policy": policy, "synthetic": args.synthetic,
        "expert_scheduling": args.expert_scheduling,
        "completed_tokens": latency.count, "completed_requests": completed_requests,
        "makespan_ms": result["makespan"] / args.ticks_per_ms,
        "output_tokens_per_sec": latency.count / seconds if seconds else 0,
        "completed_requests_per_sec": completed_requests / seconds if seconds else 0,
        "equivalent_requests_per_sec": latency.count / args.output_tokens / seconds if seconds else 0,
        "itl_count": itl.count, "itl_mean_ms": itl_summary["mean"], "itl_p95_ms": itl_summary["p95"],
        "itl_p99_ms": itl_summary["p99"], "itl_max_ms": itl_summary["max"],
        "avg_request_latency_ms": result.get("avg_request_latency_ms") if completed_requests else None,
        "avg_per_expert_batch_size": result.get("avg_per_expert_batch_size"),
        "stopped_early": result["stopped_early"],
        "token_csv": str(token_path),
    }
    measures = result.get("metrics_ms", {})
    for name in ("attention_ready_wait", "finalize_ready_wait", "expert_ready_wait", "join_wait"):
        summary[f"{name}_mean_ms"] = measures.get(name, {}).get("mean")
    busy = result.get("gpu_compute_busy_fraction")
    summary["mean_gpu_compute_busy_fraction"] = sum(busy) / len(busy) if busy else None
    # Legacy APIs may return NaN when early termination completes no requests.
    # Normalize the CSV summary too, so missing measurements stay blank.
    summary = _json_safe(summary)
    # Per-token completion records already live in the streaming CSV.
    result.pop("completion_times", None)
    _write_json(output_dir / f"{label}.json", {
        "configuration": vars(args), "provenance": provenance, "summary": summary,
        "itl_ms": itl_summary, "token_execution_latency_ms": token_summary, "simulation": result,
    })
    print(f"{label}: {latency.count} tokens, {completed_requests} requests, "
          f"{summary['output_tokens_per_sec']:.3f} tokens/s, ITL P95={itl_summary['p95']} ms")
    return summary, (label, itl.values, itl.count)


def _plot_cdf(series, output_dir, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, values, count in series:
        values = sorted(values)
        if values:
            ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post",
                    label=f"{label} ({len(values)}/{count} intervals)")
    ax.set(xlabel="ITL (ms), token_index > 0", ylabel="Empirical CDF", ylim=(0, 1),
           title="Synthetic fixture: ITL" if synthetic else "Profile-driven simulation: ITL")
    ax.grid(alpha=0.3)
    if any(values for _, values, _ in series):
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "itl_cdf.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        overhead_config = overhead_config_from_args(args)
        overhead_description = BatchOverheadModel(
            ticks_per_millisecond=args.ticks_per_ms, **overhead_config).describe()
    except ValueError as exc:
        parser.error(str(exc))
    if args.experts % args.gpus or args.top_k > args.experts:
        parser.error("--experts must divide evenly across --gpus; --top-k must not exceed --experts")
    if args.weight_decay > 1:
        parser.error("--weight-decay must be between 0 and 1")
    if args.mode != "async" and args.policies is not None:
        parser.error("--policies applies only to --mode async")
    if args.mode != "async" and args.overhead_model == "structural":
        parser.error("structural overhead needs buffer provenance; use simulation.steady_experiment "
                     "--mode sync|tbo or sim_async.run_simulation(execution_mode=...) for shared baselines")
    if args.mode != "async" and args.expert_scheduling != "layer":
        parser.error("--expert-scheduling expert applies only to --mode async")
    if args.queue_trace and args.mode != "async":
        parser.error("--queue-trace applies only to --mode async")
    if args.queue_trace_end_ms is not None and args.queue_trace_end_ms <= args.queue_trace_start_ms:
        parser.error("--queue-trace-end-ms must exceed --queue-trace-start-ms")
    policies = (args.policies or "defrag").split(",") if args.mode == "async" else [args.mode]
    policies = [policy.strip() for policy in policies]
    if args.mode == "async" and any(policy not in ("defrag", "flfs", "mtfs", "fifo") for policy in policies):
        parser.error("--policies accepts only defrag,flfs,mtfs,fifo")
    if len(set(policies)) != len(policies):
        parser.error("--policies must not contain duplicates")
    if args.synthetic:
        if args.routing_profile or args.expert_profile or args.attention_profile or args.grouped_profile:
            parser.error("--synthetic cannot be combined with supplied profiles")
        if args.attention_batch_cap is None:
            args.attention_batch_cap = 16
        if args.expert_batch_cap is None:
            args.expert_batch_cap = 64
    elif args.routing_profile is None or args.expert_profile is None:
        parser.error("provide --routing-profile and --expert-profile, or explicitly use --synthetic")
    elif args.attention_batch_cap is None or args.expert_batch_cap is None:
        parser.error("supplied profiles require explicit --attention-batch-cap and --expert-batch-cap "
                     "from the target runtime configuration; use simulation.steady_experiment "
                     "for runtime-referenced research defaults")
    for path in (args.routing_profile, args.expert_profile, args.attention_profile, args.grouped_profile):
        if path is not None and not path.is_file():
            parser.error(f"profile does not exist: {path}")

    output_dir = (args.output_dir or Path("simulation-results") /
                  datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        parser.error(f"output directory must be absent or empty; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    routing, expert = (_synthetic_profiles(args, output_dir) if args.synthetic
                       else (args.routing_profile.resolve(), args.expert_profile.resolve()))
    provenance = {
        "input_kind": "synthetic_fixture" if args.synthetic else "provided_profiles_unverified_provenance",
        "hardware_calibrated": False,
        "scheduling": _scheduling_provenance(args.expert_scheduling),
        "routing": _file_record(routing), "expert": _file_record(expert),
        "attention": _file_record(args.attention_profile), "grouped": _file_record(args.grouped_profile),
        "cost_assumptions": {
            "batch_overhead": overhead_description,
            "expert": "exact_grouped_vector_lookup" if args.grouped_profile else "sum_per_expert_profile_proxy",
            "attention": "decode_batch_context_profile" if args.attention_profile else "linear_per_token_proxy",
            "network": "assumed_latency_plus_bytes_over_bandwidth_single_host_receive_fifo",
            "final_attention": "unmeasured_linear_finalize_cost",
            "synthetic_expert_formula_ms": "0.01 + 0.002 * batch_size" if args.synthetic else None,
        },
        "units": {"arrival_rate": "requests/second", "network_bandwidth_gbps": "gigabits/second",
                  "profile_cost": "milliseconds", "time": f"{args.ticks_per_ms} ticks/millisecond"},
        "metric_scope": "finite decode-only workload; ITL is adjacent completion gap, excluding token_index=0",
        "comparison_invariants": "same routing, compute profiles, overhead assumptions, topology, arrival seed and workload across policies",
    }
    _write_json(output_dir / "inputs.json", {"configuration": vars(args), "provenance": provenance})
    summaries, series = [], []
    for policy in policies:
        summary, curve = _run_one(args, policy, routing, expert, output_dir, provenance)
        summaries.append(summary)
        series.append(curve)
    with (output_dir / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if args.plot_itl_cdf:
        _plot_cdf(series, output_dir, args.synthetic)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
