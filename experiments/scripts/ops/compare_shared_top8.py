"""Compare two uniform decode runs and report the scope of their shared Top-8 routing audit."""

import argparse
import csv
import json
import math
import re
import sys
from bisect import bisect_left
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "rtx4090"))
sys.path.insert(0, str(SCRIPTS / "sglang"))
from plot_scheduler_choices import load_run, outstanding_steps
from bench_uniform import check_decode_mode, percentiles


def metrics(submits, finishes, duration, tokens, window_tokens, window):
    lo, hi = window
    if not 0 <= lo < hi <= duration:
        raise ValueError(f"Window {window} exceeds measured run duration {duration}")
    submits, finishes = sorted(submits), sorted(finishes)
    a, b = (bisect_left(submits, t) for t in window)
    c, d = (bisect_left(finishes, t) for t in window)
    n = len(submits)
    return {
        "whole": {"duration_s": duration, "completed": n, "output_tokens": tokens,
                  "output_tok_s": tokens / duration, "completed_req_s": n / duration,
                  "e2e_ms": {"mean": 1000 * (math.fsum(finishes) - math.fsum(submits)) / n,
                             "p50": None, "p99": None}},
        "window": {"submitted": b - a, "completed": d - c, "outstanding_start": a - c,
                   "outstanding_end": b - d, "output_tokens": window_tokens,
                   "output_tok_s": window_tokens / (hi - lo), "completed_req_s": (d - c) / (hi - lo)},
    }


def streaminfer(path, window):
    choices = path if (path / "run.json").exists() else path / "choices"
    meta = load_run(choices)
    outstanding_steps(meta)  # Check that the observed request backlog never goes negative.
    cfg = meta["configuration"]
    n, prompt, output = cfg["num_requests"], cfg["min_input_len"], cfg["min_output_len"]
    if (n != len(meta["submitted_s"]) or min(prompt, output) <= 0 or cfg["max_input_len"] != prompt + 1
            or cfg["max_output_len"] != output + 1 or cfg["topk"] != 8
            or cfg.get("checkpoint_path") or cfg.get("decode_snapshot")):
        raise ValueError("Expected a fixed-length synthetic StreamInfer Top-8 run")
    start, end = meta["start_ns"], meta["end_ns"]
    lo, hi = (start + round(t * 1e9) for t in window)
    total = in_window = 0
    for gpu in meta["device_ids"]:
        with (choices / f"choices-gpu-{gpu}.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row["stage"] == "finalize":
                    stamp, size = int(row["time_ns"]), int(row["batch_size"])
                    if size <= 0 or int(row["layer"]) != meta["num_layers"] or not start <= stamp < end:
                        raise ValueError(f"Invalid GPU {gpu} Finalize dequeue")
                    total += size
                    if lo <= stamp < hi:
                        in_window += size
    if total != n * output:
        raise ValueError(f"Finalize produced {total} tokens, expected {n * output}")
    sampler = choices.parent / f"rate={meta['rate']}" / "sampler_step.csv"
    if sampler.is_file():
        with sampler.open() as handle:
            if sum(int(row["num_tokens"]) for row in csv.DictReader(handle)) != total:
                raise ValueError("StreamInfer detokenizer count differs from Finalize")
    result = metrics(meta["submitted_s"], meta["finished_s"], (end - start) / 1e9, total, in_window, window)
    result.update(token_observation="Worker host timestamp at Finalize dequeue; batch size counts tokens.",
                  latency_observation="Driver submit to response. Completion timestamps lack request IDs.",
                  sampler_total_verified=sampler.is_file())
    return (n, meta["rate"], prompt, output), result, meta["num_layers"], choices.parent


def sglang(path, window):
    report = json.loads(path.read_text())
    if (report["format"] != "sglang.uniform_http.v1" or report["status"] != "complete"
            or report["mode"] != "decode-dummy" or report["excluded_bootstrap_tokens_per_request"] != 1):
        raise ValueError("Expected a completed decode-dummy SGLang run")
    check_decode_mode(report["server_info"], True)
    rows = report["requests"]
    n, output = report["num_requests"], report["output_tokens_per_request"]
    if (n <= 0 or output <= 0 or sorted(r["index"] for r in rows) != list(range(n))
            or len({r["bootstrap_room"] for r in rows}) != n):
        raise ValueError("Invalid request indices, lengths or bootstrap rooms")
    lo, hi = window
    total = in_window = 0
    submits, finishes, latencies, window_latencies = [], [], [], []
    duration = report["whole"]["end_s"]
    for row in rows:
        a, b, times = row["submitted_s"], row["response_s"], row["token_times_s"]
        if (row["api_completion_tokens"] != output + 1 or len(times) != output
                or not 0 <= a <= row["bootstrap_token_s"] <= times[0] <= times[-1] <= b <= duration
                or any(not x <= y for x, y in zip(times, times[1:]))):
            raise ValueError(f"Invalid token accounting or timestamps on SGLang request {row['index']}")
        total += len(times)
        in_window += bisect_left(times, hi) - bisect_left(times, lo)
        submits.append(a)
        finishes.append(b)
        latency_ms = (b - a) * 1000
        latencies.append(latency_ms)
        if lo <= a < hi:
            window_latencies.append(latency_ms)
    if (report["api_completion_tokens"] != total + n or report["whole"]["completed"] != n
            or report["whole"]["output_tokens"] != total):
        raise ValueError("Global SGLang report and per-request counts disagree")
    result = metrics(submits, finishes, duration, total, in_window, window)
    result["whole"]["e2e_ms"] = percentiles(latencies)
    result["window"]["arrival_cohort_e2e_ms"] = percentiles(window_latencies)
    result.update(token_observation="HTTP client receives streaming token increments; coalesced tokens share a timestamp.",
                  latency_observation="HTTP client submit to end of stream, paired by request.",
                  excluded_bootstrap_tokens=n, api_completion_tokens=total + n)
    return (n, report["rate"], report["input_tokens"], output), result


def audit(path, dimensions, directories):
    cert = json.loads(path.read_text())
    extent, digest = cert["checked_extent"], cert["trace_sha256"]
    if (not re.fullmatch("[0-9a-f]{64}", digest) or len(extent) != len(dimensions)
            or any(type(x) is not int or not 0 < x <= y for x, y in zip(extent, dimensions))):
        raise ValueError("Invalid routing certificate extent or SHA-256")
    count = math.prod(extent)
    links = {}
    for name, directory in directories.items():
        if cert[name]["matched_unique_routes"] != count or not cert[name]["files"]:
            raise ValueError(f"{name} has incomplete routing audit coverage")
        sidecar = directory / "route-table.sha256"
        if not sidecar.is_file() or sidecar.read_text().split()[:1] != [digest]:
            raise ValueError(f"{sidecar} must contain the routing certificate SHA-256")
        links[name] = str(sidecar)
    return {"trace_sha256": digest, "axes": ["request", "decode_step", "layer"],
            "checked_extent": extent, "matched_coordinates_per_engine": count,
            "timed_workload_extent": dimensions,
            "verified_fraction": count / math.prod(dimensions),
            "timed_run_sha256_sidecars": links,
            "scope": "Only the listed audit extent was checked in both audit logs. "
                     "Both timed run SHA-256 records match the certificate. Unobserved coordinates were not audited."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streaminfer", required=True, type=Path, help="Run directory or its choices directory")
    parser.add_argument("--sglang", required=True, type=Path, help="Decode-dummy result.json")
    parser.add_argument("--verification", required=True, type=Path, help="shared_top8.py verify JSON")
    parser.add_argument("--window", default="95,125", metavar="START,END", help="Half-open interval in seconds")
    parser.add_argument("--output", type=Path, help="Write JSON here (default: stdout)")
    args = parser.parse_args()
    window = tuple(map(float, args.window.split(",")))
    if len(window) != 2 or not 0 <= window[0] < window[1] < math.inf:
        parser.error("--window requires finite, ordered START,END")
    workload, si, layers, root = streaminfer(args.streaminfer, window)
    sg_workload, sg = sglang(args.sglang, window)
    n, rate, prompt, output = workload
    if workload != sg_workload or window[1] > n / rate:
        raise ValueError(f"Unequal workloads or window outside offered interval: {workload}, {sg_workload}, {window}")
    verification = audit(args.verification, [n, output, layers], {"streaminfer": root, "sglang": args.sglang.parent})
    result = dict(requests=n, arrival_rate_req_s=rate, input_tokens=prompt, output_tokens_per_request=output,
                  window_s=window, route_audit=verification, streaminfer=si, sglang=sg)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
