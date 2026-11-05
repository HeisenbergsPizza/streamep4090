"""Classify finite sustained-arrival observations and plot empirical capacity scans.

The rules below are configurable experimental resolution criteria, not a proof
of queue stability or a mathematical throughput upper bound.  A request cap
can hide overload inside the runtime; external waiting is explicitly included.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    block_ms: float = 5000.
    relative_drift: float = .02
    absolute_drift_rps: float = .25
    min_blocks: int = 3
    confirmation_seeds: int = 3
    confirmation_arrival_ms: float = 40000.
    confirmation_warmup_ms: float = 10000.
    confirmation_measurement_ms: float = 30000.

    def validate(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.block_ms <= 0 or self.min_blocks < 3 or self.confirmation_seeds < 3:
            raise ValueError("block_ms must be positive; min_blocks and confirmation_seeds must be >= 3")
        if (self.confirmation_arrival_ms < 40000 or self.confirmation_warmup_ms < 10000
                or self.confirmation_measurement_ms < 30000):
            raise ValueError("confirmation requires at least 40 s arrivals, 10 s warmup and 30 s measurement")


def _finite(value, name, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _slope(times_seconds, values):
    if len(values) < 2:
        return None
    x = np.asarray(times_seconds, dtype=float)
    x -= x.mean()
    denominator = float(x @ x)
    return float(x @ (np.asarray(values) - np.mean(values)) / denominator) if denominator else None


def assessment_blocks(timeline, start_ms, end_ms, block_ms):
    """Reaggregate complete, aligned bins using their actual time integrals."""
    if end_ms <= start_ms or block_ms <= 0:
        raise ValueError("assessment window and block size must be positive")
    count = int(math.floor((end_ms - start_ms) / block_ms + 1e-9))
    blocks = []
    for index in range(count):
        first, last = start_ms + index * block_ms, start_ms + (index + 1) * block_ms
        selected = [item for item in timeline
                    if item["end_ms"] > first + 1e-7 and item["start_ms"] < last - 1e-7]
        if not selected:
            raise ValueError("timeline does not cover the complete assessment window")
        cursor = first
        areas = {key: 0. for key in ("active_requests", "waiting_requests", "outstanding_requests")}
        for item in selected:
            lo = _finite(item["start_ms"], "bin start")
            hi = _finite(item["end_ms"], "bin end")
            if hi <= lo or abs(lo - cursor) > 1e-6 or hi > last + 1e-6:
                raise ValueError("assessment boundaries must match contiguous complete timeline bins")
            if not item.get("coverage_complete", False):
                raise ValueError("timeline contains incomplete observed coverage")
            state = item["state_mean"]
            for key in areas:
                areas[key] += _finite(state[key], key) * (hi - lo)
            if not math.isclose(state["outstanding_requests"], state["active_requests"] + state["waiting_requests"], abs_tol=1e-6):
                raise ValueError("outstanding state must equal active plus waiting")
            cursor = hi
        if abs(cursor - last) > 1e-6:
            raise ValueError("timeline ends before assessment block boundary")
        blocks.append({"start_ms": first, "end_ms": last,
                       **{key: value / block_ms for key, value in areas.items()}})
    return blocks


def classify_blocks(blocks, observed_arrival_rps, thresholds=Thresholds()):
    """Use one backlog-drift evidence family, with external waiting validation."""
    thresholds.validate()
    epsilon = max(thresholds.absolute_drift_rps, thresholds.relative_drift * observed_arrival_rps)
    result = {"classification": "inconclusive", "drift_tolerance_rps": epsilon,
              "assessment_blocks": len(blocks), "reason": "insufficient_complete_blocks"}
    if len(blocks) < thresholds.min_blocks:
        return result
    times = [(item["start_ms"] + item["end_ms"]) / 2000 for item in blocks]
    backlog = [item["outstanding_requests"] for item in blocks]
    waiting = [item["waiting_requests"] for item in blocks]
    tail = max(2, math.ceil(len(blocks) / 2))
    span = times[-1] - times[0]
    b_slope, w_slope = _slope(times, backlog), _slope(times, waiting)
    b_end = (backlog[-1] - backlog[0]) / span
    w_end = (waiting[-1] - waiting[0]) / span
    b_tail, w_tail = _slope(times[-tail:], backlog[-tail:]), _slope(times[-tail:], waiting[-tail:])
    result.update(backlog_slope_rps=b_slope, backlog_endpoint_slope_rps=b_end,
                  backlog_tail_slope_rps=b_tail, waiting_slope_rps=w_slope,
                  waiting_endpoint_slope_rps=w_end, waiting_tail_slope_rps=w_tail,
                  first_block_waiting_mean=waiting[0], last_block_waiting_mean=waiting[-1],
                  assessment_start_ms=blocks[0]["start_ms"], assessment_end_ms=blocks[-1]["end_ms"])
    if all(abs(value) <= epsilon + 1e-10 for value in (b_slope, b_end, b_tail, w_slope, w_tail)):
        result.update(classification="stable", reason="backlog_and_waiting_drift_within_resolution")
    elif b_slope > epsilon and b_tail > epsilon and w_end > epsilon:
        result.update(classification="overload", reason="persistent_backlog_growth_with_growing_external_waiting")
    else:
        result["reason"] = "transient_or_borderline_drift_requires_longer_observation"
    return result


_COMPARISON_KEYS = (
    "gpus", "num_gpus", "num_nodes", "gpus_per_node", "layers", "num_layers", "experts", "num_experts",
    "top_k", "routing_top_k", "hidden_dim", "bytes_per_element", "output_tokens", "tokens_per_request",
    "active_request_cap", "global_request_max_batch_size", "attention_batch_cap", "expert_batch_cap",
    "max_batch_size", "initial_context", "initial_context_length", "attention_per_token_us",
    "finalize_per_token_us", "expert_cost_scale", "network_bandwidth_gbps", "network_latency_us",
    "lookahead_steps", "lookback_steps", "weight_decay", "expert_profile", "attention_profile", "grouped_profile",
)


def _comparison_identity(configuration):
    supplied = configuration.get("comparison_configuration")
    common = dict(supplied) if supplied is not None else {key: configuration[key] for key in _COMPARISON_KEYS if key in configuration}
    common["expert_scheduling"] = _expert_scheduling(configuration)
    identity = {"common": common, "warmup_ms": configuration["warmup_ms"],
                "arrival_end_ms": configuration["arrival_end_ms"]}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    physical = hashlib.sha256(json.dumps(common, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return digest, physical, "explicit_comparison_configuration" if supplied is not None else "known_common_parameter_keys"


def _expert_scheduling(configuration):
    common = configuration.get("comparison_configuration") or {}
    mode = configuration.get("expert_scheduling", common.get("expert_scheduling", "layer"))
    if mode not in ("layer", "expert"):
        raise ValueError("configuration.expert_scheduling must be layer or expert")
    if "expert_scheduling" in common and common["expert_scheduling"] != mode:
        raise ValueError("expert_scheduling disagrees with comparison_configuration")
    return mode


def _metric(window, metric, statistic="mean"):
    value = window.get("metrics_ms", {}).get(metric, {}).get(statistic)
    return None if value is None else _finite(value, f"{metric}.{statistic}")


def _completion_validation(document, steady, path):
    """Keep failed runs diagnostic without treating their windows as confirmation."""
    reasons = []
    if steady.get("fully_drained") is not True:
        reasons.append("steady_metrics_not_fully_drained")
    if "validation" in document:
        validation = document["validation"]
        if not isinstance(validation, dict) or validation.get("fully_drained") is not True:
            reasons.append("result_validation_not_fully_drained")
    status_path = path.with_name("status.json")
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if not isinstance(status, dict) or status.get("status") != "complete" or status.get("complete") is not True:
            reasons.append("status_not_complete")
    return reasons


def load_case(path, thresholds=Thresholds()):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    configuration = document["configuration"]
    steady = document["simulation"]["steady_metrics"]
    if steady.get("schema_version") != 1 or steady.get("time_unit") != "ms":
        raise ValueError("unsupported steady_metrics schema or time unit")
    policy = configuration["policy"]
    if not isinstance(policy, str) or not policy:
        raise ValueError("configuration.policy must be a nonempty string")
    expert_scheduling = _expert_scheduling(configuration)
    strategy = policy if expert_scheduling == "layer" else f"{policy}/expert"
    rate = _finite(configuration["rate_rps"], "rate_rps")
    if rate <= 0:
        raise ValueError("sustained-arrival rate_rps must be positive")
    seed = configuration["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    start = _finite(configuration["warmup_ms"], "warmup_ms")
    end = _finite(configuration["arrival_end_ms"], "arrival_end_ms")
    if end <= start:
        raise ValueError("measurement end must exceed warmup")
    windows = steady["windows"]
    matches = [(name, item) for name, item in windows.items()
               if math.isclose(item["start_ms"], start, abs_tol=1e-6)
               and math.isclose(item["end_ms"], end, abs_tol=1e-6)]
    if not matches:
        raise ValueError("no measurement window matches warmup_ms through arrival_end_ms")
    label, window = next(((name, item) for name, item in matches if name == "measurement"), matches[0])
    if not window.get("coverage_complete", False) or not math.isclose(window["observed_duration_ms"], end - start, abs_tol=1e-6):
        raise ValueError("measurement coverage is incomplete; throughput denominator must not shrink")
    counts = window["counts"]
    arrivals = _finite(counts["arrived_requests"], "arrived_requests")
    completions = _finite(counts["completed_requests"], "completed_requests")
    tokens = _finite(counts["completed_tokens"], "completed_tokens")
    if any(not value.is_integer() for value in (arrivals, completions, tokens)):
        raise ValueError("request/token event counts must be integers")
    duration = (end - start) / 1000
    initial, final = window["state_at_start"], window["state_at_end"]
    for state in (initial, final):
        for key in ("active_requests", "waiting_requests", "outstanding_requests"):
            _finite(state[key], key)
        if not math.isclose(state["outstanding_requests"], state["active_requests"] + state["waiting_requests"], abs_tol=1e-6):
            raise ValueError("boundary outstanding state must equal active plus waiting")
    change = final["outstanding_requests"] - initial["outstanding_requests"]
    if not math.isclose(change, arrivals - completions, abs_tol=1e-6):
        raise ValueError("request conservation failed: delta outstanding != arrivals - completions")
    timeline = steady["timeline"]
    blocks = assessment_blocks(timeline, start, end, thresholds.block_ms)
    classified = classify_blocks(blocks, arrivals / duration, thresholds)
    duration_eligible = (end >= thresholds.confirmation_arrival_ms - 1e-6
                         and start >= thresholds.confirmation_warmup_ms - 1e-6
                         and end - start >= thresholds.confirmation_measurement_ms - 1e-6)
    completion_reasons = _completion_validation(document, steady, path)
    eligible = duration_eligible and not completion_reasons
    signature, physical_signature, signature_scope = _comparison_identity(configuration)
    busy = window.get("gpu_busy_fraction", [])
    busy_values = list(busy.values()) if isinstance(busy, dict) else busy
    means = window.get("batch_mean", {})
    row = {"policy": strategy, "base_policy": policy, "expert_scheduling": expert_scheduling,
           "rate_rps": rate, "seed": seed, "result_path": str(path),
           "measurement_label": label, "warmup_ms": start, "arrival_end_ms": end,
           "measurement_duration_ms": end - start, "confirmation_eligible": eligible,
           "confirmation_duration_eligible": duration_eligible,
           "completion_validation_passed": not completion_reasons,
           "confirmation_ineligibility_reasons": completion_reasons + ([] if duration_eligible else ["insufficient_duration"]),
           "comparison_signature": signature, "physical_signature": physical_signature,
           "comparison_signature_scope": signature_scope,
           "source_kind": configuration.get("source_kind"), "routing_pool_size": configuration.get("routing_pool_size"),
           "arrived_requests": int(arrivals), "completed_requests": int(completions), "completed_tokens": int(tokens),
           "observed_arrival_rps": arrivals / duration, "completed_requests_per_sec": completions / duration,
           "output_tokens_per_sec": tokens / duration,
           "completion_to_arrival_ratio": completions / arrivals if arrivals else None,
           "net_backlog_growth_rps": change / duration,
           "active_requests_time_mean": window["state_mean"]["active_requests"],
           "waiting_requests_time_mean": window["state_mean"]["waiting_requests"],
           "outstanding_requests_time_mean": window["state_mean"]["outstanding_requests"],
           "waiting_requests_at_measurement_end": final["waiting_requests"],
           "outstanding_requests_at_measurement_end": final["outstanding_requests"],
           "oldest_waiting_age_at_measurement_end_ms": (window.get("ages_at_end_ms") or {}).get("oldest_waiting_request"),
           "oldest_unfinished_age_at_measurement_end_ms": (window.get("ages_at_end_ms") or {}).get("oldest_unfinished_request"),
           "remaining_output_tokens_at_measurement_end": final.get("remaining_output_work"),
           "attention_batch_mean": means.get("attention"), "expert_group_batch_mean": means.get("expert_group"),
           "expert_individual_batch_mean": means.get("expert_individual"),
           "itl_mean_ms": _metric(window, "itl"), "itl_p95_ms": _metric(window, "itl", "p95"),
           "admission_wait_mean_ms": _metric(window, "admission_wait"),
           "admission_wait_p95_ms": _metric(window, "admission_wait", "p95"),
           "request_latency_p95_ms": _metric(window, "request_latency", "p95"),
           "attention_ready_wait_mean_ms": _metric(window, "attention_ready_wait"),
           "expert_ready_wait_mean_ms": _metric(window, "expert_ready_wait"),
           "mean_gpu_compute_busy_fraction": float(np.mean(busy_values)) if len(busy_values) else None,
           "fully_drained": steady.get("fully_drained", False), **classified}
    return {"summary": row, "configuration": configuration, "steady_metrics": steady,
            "measurement": window, "blocks": blocks}


def select_representative_cases(cases):
    """Choose once per strategy/rate/seed for every CSV, plot and bracket consumer.

    Prefer a validated run, then confirmation duration, then a longer measured
    window and arrival horizon. Equal-priority duplicates use path order so
    discovery order cannot change the result. Every discarded source is recorded.
    """
    grouped = defaultdict(list)
    for case in cases:
        row = case["summary"]
        grouped[(row["policy"], row["rate_rps"], row["seed"])].append(case)

    def priority(case):
        row = case["summary"]
        return (row["completion_validation_passed"], row["confirmation_duration_eligible"],
                row["measurement_duration_ms"], row["arrival_end_ms"])

    selected, discarded = [], []
    for key, group in sorted(grouped.items()):
        group = sorted(group, key=lambda case: case["summary"]["result_path"])
        chosen = max(group, key=priority)
        selected.append(chosen)
        for case in group:
            if case is chosen:
                continue
            row = case["summary"]
            discarded.append({"policy": key[0], "rate_rps": key[1], "seed": key[2],
                              "result_path": row["result_path"],
                              "selected_result_path": chosen["summary"]["result_path"],
                              "reason": "equal_priority_path_tiebreak" if priority(case) == priority(chosen)
                                        else "less_sufficient_observation",
                              "measurement_duration_ms": row["measurement_duration_ms"],
                              "arrival_end_ms": row["arrival_end_ms"],
                              "confirmation_eligible": row["confirmation_eligible"],
                              "completion_validation_passed": row["completion_validation_passed"],
                              "physical_configuration_changed": row["physical_signature"] != chosen["summary"]["physical_signature"]})
    return selected, discarded


def capacity_brackets(cases, thresholds=Thresholds()):
    """Report confirmed empirical brackets, preserving mixed/nonmonotone scans."""
    cases, _ = select_representative_cases(cases)
    grouped = defaultdict(lambda: defaultdict(list))
    for case in cases:
        row = case["summary"]
        grouped[row["policy"]][row["rate_rps"]].append(row)
    results = {}
    for policy, rate_groups in sorted(grouped.items()):
        points = []
        for rate, chosen in sorted(rate_groups.items()):
            classes = {row["classification"] for row in chosen}
            signatures = {row["comparison_signature"] for row in chosen}
            eligible = [row for row in chosen if row["confirmation_eligible"]]
            classification = next(iter(classes)) if len(classes) == 1 else "inconclusive"
            mixed = len(classes) > 1
            if len(signatures) != 1:
                status = "mixed_configuration"
            elif mixed:
                status = "mixed_seeds"
            elif classification == "inconclusive":
                status = "inconclusive"
            elif any(not row["completion_validation_passed"] for row in chosen):
                status = "unvalidated_completion"
            elif len(eligible) >= thresholds.confirmation_seeds and len(eligible) == len(chosen):
                status = "confirmed"
            else:
                status = "screen_only"
            points.append({"rate_rps": rate, "classification": classification, "confirmation": status,
                           "seed_count": len(chosen), "eligible_seed_count": len(eligible),
                           "seeds": [{"seed": row["seed"], "classification": row["classification"],
                                      "eligible": row["confirmation_eligible"], "result_path": row["result_path"]}
                                     for row in sorted(chosen, key=lambda item: item["seed"])],
                           "completed_rps_mean": float(np.mean([row["completed_requests_per_sec"] for row in chosen])),
                           "completed_rps_min": min(row["completed_requests_per_sec"] for row in chosen),
                           "completed_rps_max": max(row["completed_requests_per_sec"] for row in chosen)})
        stable = [point["rate_rps"] for point in points if point["classification"] == "stable" and point["confirmation"] == "confirmed"]
        overload = [point["rate_rps"] for point in points if point["classification"] == "overload" and point["confirmation"] == "confirmed"]
        nonmonotone = any(low["rate_rps"] < high["rate_rps"] and low["classification"] == "overload" and high["classification"] == "stable"
                          for low in points for high in points)
        lower = max(stable) if stable else None
        higher = [rate for rate in overload if lower is not None and rate > lower]
        upper = min(higher) if higher else None
        reasons = []
        if not stable:
            reasons.append("no_rate_confirmed_stable_with_required_seeds_and_duration")
        if upper is None:
            reasons.append("no_higher_rate_confirmed_overloaded")
        if any(point["confirmation"] in ("mixed_seeds", "mixed_configuration") for point in points):
            reasons.append("mixed_seed_or_configuration_results")
        if any(not row["completion_validation_passed"] for rows in rate_groups.values() for row in rows):
            reasons.append("run_completion_not_validated")
        if nonmonotone:
            reasons.append("nonmonotone_observed_pattern_do_not_assume_binary_search")
        if len({row["physical_signature"] for rows in rate_groups.values() for row in rows}) > 1:
            reasons.append("different_model_or_cost_configurations_across_rates")
        results[policy] = {"status": "unresolved" if reasons else "empirical_bracket",
                           "stable_lower_rate_rps": lower, "overload_upper_rate_rps": upper,
                           "nonmonotone_observed": nonmonotone, "reasons": reasons, "rate_points": points,
                           "interpretation": "finite-window empirical bracket at the configured drift resolution; not a mathematical upper bound"}
    return results


def _save_figures(cases, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = sorted({case["summary"]["policy"] for case in cases}, key=lambda name: ({"defrag": 0, "flfs": 1, "mtfs": 2, "fifo": 3}.get(name, 4), name))
    colors = dict(zip(policies, plt.get_cmap("tab10").colors))
    rates = sorted({case["summary"]["rate_rps"] for case in cases})
    rows = [case["summary"] for case in cases]
    heading = title if "simulation proxy" in title.lower() else f"Simulation proxy · {title}"

    def save(figure, stem):
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{stem}.{extension}", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)

    def trend(axis, key, ylabel, marker="o"):
        for policy in policies:
            values = [(rate, [row[key] for row in rows if row["policy"] == policy and row["rate_rps"] == rate
                              and row.get(key) is not None]) for rate in rates]
            values = [(rate, observed) for rate, observed in values if observed]
            if not values:
                continue
            x = [item[0] for item in values]
            means = np.array([np.mean(item[1]) for item in values])
            lo = means - [min(item[1]) for item in values]
            hi = [max(item[1]) for item in values] - means
            axis.errorbar(x, means, yerr=[lo, hi], color=colors[policy], marker=marker, markersize=4,
                          linewidth=1.3, capsize=2, label=policy)
        axis.set(xlabel="Offered arrival rate (requests/s)", ylabel=ylabel)
        axis.grid(color=".9", linewidth=.6)

    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axis = plt.subplots(figsize=(8.6, 5.0), layout="constrained")
        trend(axis, "completed_requests_per_sec", "Completed requests/s (measurement window)", marker=None)
        highest = max(max(rates), max(row["completed_requests_per_sec"] for row in rows)) * 1.05
        axis.plot([0, highest], [0, highest], linestyle="--", color=".45", linewidth=1, label="completion = offered")
        markers = {"stable": "o", "overload": "^", "inconclusive": "x"}
        for row in rows:
            axis.scatter(row["rate_rps"], row["completed_requests_per_sec"], marker=markers[row["classification"]],
                         s=32, color=colors[row["policy"]], alpha=.7)
        axis.set_xlim(0, highest)
        axis.set_ylim(bottom=0)
        axis.legend(ncol=2)
        axis.set_title(f"{heading}\nWarmup/drain excluded; error bars: observed seed range, not confidence intervals")
        figure.supxlabel("Case markers: circle = window stable; triangle = window overload; × = inconclusive", fontsize=8)
        save(figure, "throughput_vs_offered")

        figure, axes = plt.subplots(len(rates), 3, figsize=(13.5, max(3.2, 2.15 * len(rates))), squeeze=False,
                                   sharex=True, layout="constrained")
        columns = (("waiting_requests", "External waiting requests"), ("active_requests", "Active requests"),
                   ("outstanding_requests", "Outstanding = active + waiting"))
        for row_index, rate in enumerate(rates):
            group = [case for case in cases if case["summary"]["rate_rps"] == rate]
            for column, (key, label) in enumerate(columns):
                axis = axes[row_index, column]
                maximum = 0.
                for case in group:
                    series = [item for item in case["steady_metrics"]["timeline"]
                              if item["end_ms"] > item["start_ms"] and item.get("state_mean") is not None]
                    if not series:
                        continue
                    ends = [series[0]["start_ms"] / 1000] + [item["end_ms"] / 1000 for item in series]
                    values = [item["state_mean"][key] for item in series]
                    maximum = max(maximum, max(values, default=0))
                    axis.stairs(values, ends, color=colors[case["summary"]["policy"]], linewidth=.85, alpha=.65)
                for boundary in sorted({case["summary"]["warmup_ms"] / 1000 for case in group}):
                    axis.axvline(boundary, color=".6", linewidth=.7, linestyle=":")
                for boundary in sorted({case["summary"]["arrival_end_ms"] / 1000 for case in group}):
                    axis.axvline(boundary, color=".35", linewidth=.7, linestyle="--")
                axis.set_ylim(bottom=0)
                if maximum == 0:
                    axis.set_ylim(0, 1)
                    axis.text(.5, .5, "Zero throughout recorded window", transform=axis.transAxes,
                              ha="center", va="center", color=".45", fontsize=8)
                axis.grid(axis="y", color=".9", linewidth=.6)
                if row_index == 0:
                    axis.set_title(label)
                if column == 0:
                    axis.set_ylabel(f"{rate:g} req/s\nRequests")
                if row_index == len(rates) - 1:
                    axis.set_xlabel("Simulation time (s)")
        handles = [plt.Line2D([], [], color=colors[policy], label=policy) for policy in policies]
        axes[0, -1].legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1.14), ncol=min(4, len(policies)))
        figure.suptitle(f"{heading} · Full observed timeline\nEach line is one case/seed; dotted = warmup end, dashed = arrival end", fontsize=11)
        save(figure, "backlog_timeline")

        figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), layout="constrained")
        metrics = (("attention_batch_mean", "Mean Attention batch"), ("expert_group_batch_mean", "Mean Expert-group batch (copies)"),
                   ("expert_individual_batch_mean", "Mean batch per nonempty expert"), ("itl_p95_ms", "ITL P95 (ms)"),
                   ("expert_ready_wait_mean_ms", "Mean expert-branch ready wait (ms)"),
                   ("mean_gpu_compute_busy_fraction", "Mean GPU compute busy fraction"))
        for axis, (key, label) in zip(axes.flat, metrics):
            trend(axis, key, label)
            axis.set_ylim(bottom=0)
        axes[0, 0].legend(ncol=2)
        figure.suptitle(f"{heading} · Scheduler observations\nMeasurement windows only; error bars show observed seed range", fontsize=11)
        save(figure, "scheduler_metrics")


def analyze(root, output_dir, thresholds=Thresholds(), title="Sustained-arrival scheduler study"):
    thresholds.validate()
    root, output = Path(root).resolve(), Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output directory must be absent or empty: {output}")
    paths = sorted(root.rglob("result.json"))
    if not paths:
        raise ValueError(f"no result.json cases found under {root}")
    cases = []
    for path in paths:
        try:
            cases.append(load_case(path, thresholds))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid case {path}: {error}") from error
    input_case_count = len(cases)
    cases, discarded = select_representative_cases(cases)
    output.mkdir(parents=True, exist_ok=True)
    rows = [case["summary"] for case in cases]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "case_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    policies = capacity_brackets(cases, thresholds)
    report = {"format": "streaminfer.steady_analysis", "version": 1, "input_root": str(root),
              "thresholds": asdict(thresholds), "case_count": len(cases), "policies": policies,
              "input_case_count": input_case_count, "discarded_case_count": len(discarded),
              "case_selection": "one strategy/rate/seed case; strategy includes expert scheduling granularity; prefer validated completion, confirmation duration, longer measurement, longer arrival horizon, then ascending absolute path",
              "discarded_cases": discarded,
              "classification_semantics": {
                  "stable": "full/endpoint/tail backlog drift and full/tail external-waiting drift within epsilon",
                  "overload": "full and tail backlog slopes exceed epsilon and external waiting endpoint drift exceeds epsilon",
                  "epsilon": "max(absolute_drift_rps, relative_drift * observed_arrival_rps)",
                  "backlog": "outstanding requests = active + external waiting",
                  "evidence": "backlog slopes and arrival-completion deficits are one conservation-related evidence family, not independent votes",
                  "confirmation": "required distinct seeds with complete long windows, fully drained runs, complete status/validation when present, and consistent classifications/configuration",
              },
              "limitations": ["Finite windows cannot prove mathematical stability or a hardware throughput upper bound.",
                              "Small drift within the configured resolution can remain undetected; extend gray/boundary cases.",
                              "Mixed seeds and nonmonotone batching behavior remain unresolved; do not blindly binary-search.",
                              "Arrival-window throughput excludes drain; full timeline may include drain or explicitly recorded idle coverage.",
                              "Completion/queue-delay samples omit still-unfinished waits; inspect external waiting and outstanding state.",
                              "Plots show observed case/seed ranges, not confidence intervals.",
                              "Routing may bootstrap a finite pool of request prefixes; inference is conditional on that pool, not all model traffic."]}
    (output / "capacity_brackets.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _save_figures(cases, output, title)
    for policy, entry in policies.items():
        print(f"{policy}: {entry['status']}; stable lower={entry['stable_lower_rate_rps']}, overload upper={entry['overload_upper_rate_rps']} req/s")
        for point in entry["rate_points"]:
            seeds = ", ".join(f"{item['seed']}:{item['classification']}{'*' if item['eligible'] else ''}" for item in point["seeds"])
            print(f"  rate={point['rate_rps']:g}: {point['confirmation']}; seeds [{seeds}] (* = confirmation eligible)")
        if entry["reasons"]:
            print("  " + "; ".join(entry["reasons"]))
    if discarded:
        print(f"Selected {len(cases)} of {input_case_count} cases; {len(discarded)} discarded sources recorded in capacity_brackets.json.")
    print(f"Wrote {output}; empirical observations only, not a mathematical capacity bound.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "--root", dest="root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Sustained-arrival scheduler study")
    parser.add_argument("--block-ms", type=float, default=5000.)
    parser.add_argument("--relative-drift", type=float, default=.02)
    parser.add_argument("--absolute-drift-rps", type=float, default=.25)
    parser.add_argument("--min-blocks", type=int, default=3)
    parser.add_argument("--confirmation-seeds", type=int, default=3)
    parser.add_argument("--confirmation-arrival-ms", type=float, default=40000.)
    parser.add_argument("--confirmation-warmup-ms", type=float, default=10000.)
    parser.add_argument("--confirmation-measurement-ms", type=float, default=30000.)
    args = parser.parse_args(argv)
    thresholds = Thresholds(**{key: getattr(args, key) for key in Thresholds.__dataclass_fields__})
    try:
        analyze(args.root, args.output_dir, thresholds, args.title)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
