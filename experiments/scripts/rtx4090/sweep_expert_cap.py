#!/usr/bin/env python3
"""Compare expert caps on the B512 CUDA Graph workload without a profiler.

Run in the streamep4090 environment. Each trial starts its own runtime and uses
the existing synchronized-start benchmark. No inference implementation changes.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[3]
LAUNCHER = Path(__file__).with_name("profile_graph_b512.sh")
METRICS = (
    "e2e_duration", "req_throughput", "token_throughput",
    "req_latency_mean_ms", "req_latency_median_ms", "req_latency_p99_ms",
    "itl_latency_mean_ms", "itl_latency_median_ms", "itl_latency_p99_ms",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caps", type=int, nargs="+", choices=(128, 256, 512, 1024),
                        default=[128, 256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=1,
                        help="Trials per cap (default: 1; use 3 to assess variability)")
    parser.add_argument("--gpu-usage", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path,
                        help="New directory for the sweep; must not already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate every trial with the real CLI; no GPU or output files")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if len(set(args.caps)) != len(args.caps):
        parser.error("--caps must not contain duplicates")
    if not 0 < args.gpu_usage < 1:
        parser.error("--gpu-usage must be between 0 and 1")
    if args.output_dir is None:
        args.output_dir = REPO / "reports/rtx4090" / (
            f"expert-cap-sweep-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
        )
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def make_plan(args):
    plan = []
    for repeat in range(1, args.repeats + 1):
        # Rotate which cap goes first instead of always favoring the same order.
        offset = (repeat - 1) % len(args.caps)
        order = args.caps[offset:] + args.caps[:offset]
        for cap in order:
            name = f"cap-{cap}-repeat-{repeat}"
            command = ["bash", str(LAUNCHER), "baseline",
                       "--max-batch-size-expert", str(cap),
                       "--gpu-usage", str(args.gpu_usage)]
            if args.dry_run:
                command.append("--dry-run")
            plan.append({
                "expert_cap": cap, "repeat": repeat, "command": command,
                "run_dir": str(args.output_dir / name),
                "launcher_log": str(args.output_dir / f"{name}.launcher.log"),
            })
    return plan


def session_groups(session_id):
    """Find live groups in our session, including GNU timeout's separate group."""
    listing = subprocess.run(["ps", "-e", "-o", "pid=,stat="], check=True,
                             capture_output=True, text=True).stdout
    groups = set()
    for line in listing.splitlines():
        pid_text, state = line.split(None, 1)
        if state.startswith("Z"):
            continue
        pid = int(pid_text)
        try:
            if os.getsid(pid) == session_id:
                groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    return groups


def stop_trial(process, grace_seconds=35):
    # Do not use shell completion as proof that timeout and the driver exited.
    for group in session_groups(process.pid):
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not session_groups(process.pid):
            break
        time.sleep(0.2)
    for group in session_groups(process.pid):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_one(trial, env):
    """Keep verbose output in a log; terminate only this trial on interruption."""
    with open(trial["launcher_log"], "w", encoding="utf-8") as log:
        process = subprocess.Popen(trial["command"], cwd=REPO, env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt

        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            return process.wait()
        except KeyboardInterrupt:
            stop_trial(process)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


def read_metrics(path):
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, strict=True)
            rows = list(reader)
            fields = reader.fieldnames or []
    except csv.Error as exc:
        raise ValueError(f"Malformed metrics CSV: {path}: {exc}") from exc
    if len(rows) != 1:
        raise ValueError(f"Expected one metrics row, found {len(rows)} in {path}")
    row = rows[0]
    required = {"num_requests", "DP_size", "EP_size", "num_experts", *METRICS}
    if (len(fields) != len(set(fields)) or None in row
            or any(row.get(key) in (None, "") for key in required)):
        raise ValueError(f"Missing/duplicate columns or truncated metrics row in {path}")
    for key, expected in {"num_requests": 512, "DP_size": 4,
                          "EP_size": 4, "num_experts": 128}.items():
        if float(row[key]) != expected:
            raise ValueError(f"Unexpected {key}={row[key]!r} in {path}")
    metrics = {key: float(row[key]) for key in METRICS}
    for key, value in metrics.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}={value} in {path}")
    if any(metrics[key] <= 0 for key in ("e2e_duration", "req_throughput", "token_throughput")):
        raise ValueError(f"Nonpositive duration/throughput in {path}")
    return metrics


def write_csv(path, fields, rows):
    # Publish a complete summary after every trial, including a failed trial.
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_results(output_dir, results):
    fields = ["expert_cap", "repeat", "status", "exit_code", *METRICS,
              "run_dir", "launcher_log", "error"]
    write_csv(output_dir / "runs.csv", fields, results)
    summary = []
    for cap in sorted({row["expert_cap"] for row in results}):
        trials = [row for row in results if row["expert_cap"] == cap]
        good = [row for row in trials if row["status"] == "ok"]
        row = {"expert_cap": cap, "successful_runs": len(good),
               "failed_runs": len(trials) - len(good)}
        if good:
            row.update({f"median_{key}": statistics.median(r[key] for r in good)
                        for key in METRICS})
            row["min_itl_latency_mean_ms"] = min(r["itl_latency_mean_ms"] for r in good)
            row["max_itl_latency_mean_ms"] = max(r["itl_latency_mean_ms"] for r in good)
        summary.append(row)
    fields = ["expert_cap", "successful_runs", "failed_runs",
              *(f"median_{key}" for key in METRICS),
              "min_itl_latency_mean_ms", "max_itl_latency_mean_ms"]
    write_csv(output_dir / "summary.csv", fields, summary)
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(f"Choose a new --output-dir; already exists: {args.output_dir}")
    plan = make_plan(args)
    print(f"Expert caps: {args.caps}; repeats: {args.repeats}; trials: {len(plan)}", flush=True)
    print("B512, A128, DP4/EP4/TP1, input64/output128, defrag, both CUDA Graphs ON.", flush=True)
    print("Nsight/NVTX OFF. Random weights/routing are not seeded across trials.", flush=True)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True)
        manifest = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable, "gpu_usage": args.gpu_usage,
            "repeats": args.repeats, "trials": plan,
            "runtime_tmpdir": os.environ.get("DMOE_RUNTIME_TMPDIR"),
            "ray_tmpdir": os.environ.get("DMOE_RAY_TMPDIR"),
            "note": "Synthetic decode; random weights/routing not fixed. Summary medians are across trials, not pooled token percentiles.",
        }
        (args.output_dir / "sweep.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Results: {args.output_dir}", flush=True)
    results = []
    for index, trial in enumerate(plan, 1):
        env = dict(os.environ, PYTHON=sys.executable, PROFILE_RUN_DIR=trial["run_dir"])
        print(f"[{index}/{len(plan)}] cap={trial['expert_cap']}, repeat={trial['repeat']}", flush=True)
        if args.dry_run:
            print(shlex.join(trial["command"]), flush=True)
            completed = subprocess.run(trial["command"], cwd=REPO, env=env)
            if completed.returncode:
                return completed.returncode
            continue
        row = {key: trial[key] for key in ("expert_cap", "repeat", "run_dir", "launcher_log")}
        row.update(status="failed", exit_code="", error="")
        try:
            row["exit_code"] = run_one(trial, env)
            if row["exit_code"] != 0:
                raise RuntimeError(f"Launcher exited with code {row['exit_code']}")
            row.update(read_metrics(Path(trial["run_dir"]) / "metrics.csv"))
            row["status"] = "ok"
        except KeyboardInterrupt:
            row.update(status="interrupted", exit_code=130, error="Interrupted by user")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
        summary = save_results(args.output_dir, results)
        if row["status"] != "ok":
            print(f"STOP: {row['error']}\nLog: {row['launcher_log']}\n"
                  f"Partial results: {args.output_dir / 'runs.csv'}", file=sys.stderr, flush=True)
            return 130 if row["status"] == "interrupted" else 1
        print(f"  mean ITL={row['itl_latency_mean_ms']:.3f} ms; "
              f"P99 ITL={row['itl_latency_p99_ms']:.3f} ms; "
              f"first-to-last token={row['req_latency_mean_ms']:.3f} ms; "
              f"cohort={row['e2e_duration']:.3f} s; "
              f"throughput={row['token_throughput']:.2f} token/s", flush=True)
    if args.dry_run:
        print("PASS: all trial configurations validated; no inference or output files.")
    else:
        print("\nAcross-trial medians (ITL in ms; cohort duration in seconds):")
        print(f"{'cap':>6} {'runs':>5} {'mean ITL':>12} {'P99 ITL':>12} {'cohort':>12} {'token/s':>12}")
        for row in summary:
            print(f"{row['expert_cap']:6d} {row['successful_runs']:5d} "
                  f"{row['median_itl_latency_mean_ms']:12.3f} "
                  f"{row['median_itl_latency_p99_ms']:12.3f} "
                  f"{row['median_e2e_duration']:12.3f} "
                  f"{row['median_token_throughput']:12.2f}")
        print(f"PASS: {len(results)} trials completed. Summary: {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
