"""Single-node NCCL point-to-point latency and bandwidth calibration."""
from __future__ import annotations

import csv
from datetime import timedelta
from itertools import combinations
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from time import perf_counter_ns

from .common import StageWriter, load_manifest, write_json


def transfer_cases(world):
    """Each case uses disjoint two-rank channels, matching the runtime communicator size."""
    if world < 2:
        raise ValueError("Network profiling requires two or more visible GPUs")
    for a, b in combinations(range(world), 2):
        yield "isolated", [(a, b)]
        yield "isolated", [(b, a)]
    matching = [(i, i + 1) for i in range(0, world - 1, 2)]
    if len(matching) > 1:
        yield "disjoint", matching
        yield "disjoint", [(b, a) for a, b in matching]
    yield "bidirectional", matching


def message_sizes(csv_sizes, hidden_size):
    sizes = ([int(x) for x in csv_sizes.split(",")] if csv_sizes is not None else
             [2 * hidden_size * rows for rows in (1, 4, 16, 64, 128, 512)])
    if len(set(sizes)) != len(sizes) or len(sizes) < 3 or any(x <= 0 or x % 2 for x in sizes):
        raise ValueError("Choose at least three distinct positive, even BF16 message byte counts")
    return sorted(sizes)


def fit_network(samples, tolerance):
    """Fit one receiver-service parameter pair; report error on every directed link/size."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("fit tolerance must be finite and positive")
    buckets = {}
    for row in samples:
        if row["scenario"] == "isolated" and row["role"] == "receive":
            key = int(row["src"]), int(row["dst"]), int(row["bytes"])
            time_us = float(row["endpoint_ms"]) * 1000
            if not math.isfinite(time_us) or time_us <= 0:
                raise ValueError("NCCL samples must be positive finite durations")
            buckets.setdefault(key, []).append(time_us)
    by_size = {}
    curve = []
    for (src, dst, size), values in sorted(buckets.items()):
        median = statistics.median(values)
        curve.append(dict(src=src, dst=dst, bytes=size, median_us=median))
        by_size.setdefault(size, []).append(median)
    if len(by_size) < 3:
        raise ValueError("At least three isolated message sizes are needed")
    # First give each directed link equal weight, then each size equal relative weight.
    points = [(x, statistics.median(times)) for x, times in sorted(by_size.items())]
    weights = [1 / y**2 for _, y in points]
    weight = sum(weights)
    x0 = sum(w * x for w, (x, y) in zip(weights, points)) / weight
    y0 = sum(w * y for w, (x, y) in zip(weights, points)) / weight
    beta = (sum(w * (x - x0) * (y - y0) for w, (x, y) in zip(weights, points))
            / sum(w * (x - x0)**2 for w, (x, y) in zip(weights, points)))
    report = {"method": "nonnegative intercept, relative least squares on median directed-link "
                        "receiver durations at each byte size; checked against every link/size median",
              "tolerance_relative": tolerance, "link_medians": curve}
    if beta <= 0 or len({y for _, y in points}) == 1:
        return {}, {**report, "status": "no increasing byte-time relationship"}
    alpha = y0 - beta * x0
    if alpha < 0:
        alpha = 0.0
        beta = (sum(w * x * y for w, (x, y) in zip(weights, points))
                / sum(w * x**2 for w, (x, y) in zip(weights, points)))
    max_error = 0.0
    for row in curve:
        row["predicted_us"] = alpha + beta * row["bytes"]
        row["relative_error"] = abs(row["predicted_us"] - row["median_us"]) / row["median_us"]
        max_error = max(max_error, row["relative_error"])
    # beta is us per byte, so payload throughput is 8e-3 / beta decimal gigabits/s.
    parameters = {"--network-latency-us": alpha, "--network-bandwidth-gbps": 0.008 / beta}
    report.update(candidate_parameters=parameters, max_link_relative_error=max_error)
    if max_error > tolerance:
        return {}, {**report, "status": "scalar fit exceeds error tolerance"}
    return parameters, {**report, "status": "fitted"}


def _network_worker(rank, world, rendezvous, output_name, sizes, warmup, repeats, timeout_seconds):
    # Set logging in the child before the first CUDA/NCCL operation. Preserve transport knobs
    # (P2P, SHM, IB, etc.) so the benchmark exercises the deployment's actual configuration.
    output = Path(output_name)
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,GRAPH,P2P,SHM,NET"
    os.environ["NCCL_DEBUG_FILE"] = str(output / f"nccl-rank-{rank}.log")
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    timeout = timedelta(seconds=timeout_seconds)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timeout)
    try:
        groups = {}
        for a, b in combinations(range(world), 2):
            group = dist.new_group(ranks=[a, b], backend="nccl", timeout=timeout)
            if rank in (a, b):
                groups[a, b] = group
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        send = torch.full((max(sizes) // 2,), rank + 1, dtype=torch.bfloat16, device="cuda")
        recv = torch.empty_like(send)
        fields = ["scenario", "rank", "src", "dst", "bytes", "trial", "role", "endpoint_ms",
                  "host_submit_us"]
        with (output / f"samples-rank-{rank}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for scenario, pairs in transfer_cases(world):
                pair = next((pair for pair in pairs if rank in pair), None)
                for size in sizes:
                    tx = send[:size // 2]
                    rx = recv[:size // 2]
                    if pair is not None:
                        src, dst = pair
                        peer = dst if rank == src else src
                        group = groups[tuple(sorted(pair))]
                        rx.zero_()
                    torch.cuda.synchronize()
                    for trial in range(-warmup, repeats):
                        # All ranks enter a CPU control barrier even if this GPU is idle.
                        dist.barrier()
                        if pair is None:
                            continue
                        start.record()
                        issue = perf_counter_ns()
                        if scenario == "bidirectional":
                            ops = [dist.P2POp(dist.isend, tx, peer, group),
                                   dist.P2POp(dist.irecv, rx, peer, group)]
                            works = dist.batch_isend_irecv(ops if rank == src else ops[::-1])
                            role = "send_receive"
                        elif rank == src:
                            works = [dist.isend(tx, dst=peer, group=group)]
                            role = "send"
                        else:
                            works = [dist.irecv(rx, src=peer, group=group)]
                            role = "receive"
                        submit_us = (perf_counter_ns() - issue) / 1000
                        for work in works:
                            work.wait()  # establish the NCCL completion dependency on our stream
                        end.record()
                        end.synchronize()
                        if trial >= 0:
                            duration = start.elapsed_time(end)
                            if not math.isfinite(duration) or duration <= 0:
                                raise RuntimeError("Invalid NCCL endpoint elapsed time")
                            writer.writerow(dict(scenario=scenario, rank=rank, src=src, dst=dst, bytes=size,
                                                 trial=trial, role=role, endpoint_ms=duration,
                                                 host_submit_us=submit_us))
                    if pair is not None and (scenario == "bidirectional" or rank == dst):
                        if not bool(torch.all(rx == peer + 1).item()):
                            raise RuntimeError(f"Incorrect NCCL payload received by rank {rank}")
        dist.barrier()
        # Destroy in identical creation order on both members.
        for group in groups.values():
            dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


def transport_evidence(output):
    """Collect NCCL's own negotiated-channel descriptions; topology alone cannot prove a path."""
    routes = {}
    for path in sorted(Path(output).glob("nccl-rank-*.log")):
        evidence = set()
        with path.open(errors="replace") as f:
            for line in f:
                match = re.search(r"\bChannel\s+.*?\s(?:via|using)\s+.+", line, re.I)
                if match:
                    evidence.add(match.group().strip())
        if evidence:
            routes[path.name] = sorted(evidence)
    return {"source": "NCCL INFO channel lines; raw per-rank logs included",
            "status": "observed" if routes else "transport unknown; NCCL emitted no matching channel lines",
            "channels": routes}


def run_network(args):
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    manifest = load_manifest(args.bundle)
    world = len(manifest["hardware"]["gpus"])
    sizes = message_sizes(args.bytes, manifest["model"]["hidden_size"])
    if min(args.warmup, args.repeats, args.timeout_seconds) < 1:
        raise ValueError("warmup, repeats and timeout-seconds must be positive")
    if not math.isfinite(args.fit_max_relative_error) or args.fit_max_relative_error <= 0:
        raise ValueError("fit-max-relative-error must be finite and positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run network once per node; it spawns its own local GPU workers")
    if not dist.is_available() or not dist.is_nccl_available() or not dist.is_gloo_available():
        raise RuntimeError("Network profiling needs a CUDA PyTorch build with Gloo and NCCL")
    with StageWriter(args.bundle, "network") as output:
        with tempfile.TemporaryDirectory(prefix="streaminfer-network-") as tmp:
            rendezvous = (Path(tmp) / "store").resolve().as_uri()
            mp.spawn(_network_worker, args=(world, rendezvous, str(output.resolve()), sizes,
                                           args.warmup, args.repeats, args.timeout_seconds), nprocs=world)
        rows = []
        for rank in range(world):
            with (output / f"samples-rank-{rank}.csv").open(newline="") as f:
                rows.extend(csv.DictReader(f))
        parameters, fit = fit_network(rows, args.fit_max_relative_error)
        write_json(output / "parameters.json", parameters)
        write_json(output / "details.json", {
            "device_count": world, "interconnect_from_manifest": manifest["hardware"]["interconnect"],
            "message_sizes_bytes": sizes, "warmup": args.warmup, "repeats": args.repeats,
            "scenarios": list(transfer_cases(world)), "endpoint_time_unit": "milliseconds",
            "timing_scope": "Per-rank CUDA event envelope from submission to local NCCL completion. "
                            "Includes exposed PyTorch launch delay and peer-arrival skew. Gloo barrier "
                            "precedes timing. Disjoint and full-duplex measurements diagnose contention "
                            "and do not contribute to the single-receiver service fit. NCCL operations are eager. "
                            "host_submit_us measures issuing the PyTorch send/receive API up to its return, "
                            "before work.wait(); these host and device intervals can overlap.",
            "implementation": "one process per CUDA device, two-rank ProcessGroupNCCL per device pair",
            "torch": str(torch.__version__), "torch_nccl": torch.cuda.nccl.version(),
            "transport_environment": {k: os.environ.get(k) for k in
                                      ("CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL",
                                       "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE", "NCCL_NET")},
            "cuda_peer_access_capable": [[None if a == b else torch.cuda.can_device_access_peer(a, b)
                                         for b in range(world)] for a in range(world)],
            "transport": transport_evidence(output), "fit": fit,
        })
    print(f"Network stage: {fit['status']}. See {Path(args.bundle) / 'network' / 'details.json'}")


def add_commands(subparsers):
    parser = subparsers.add_parser("network", help="Profile single-node directed NCCL sends and receives")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bytes", help="Comma-separated even message sizes; default BF16 hidden rows x 1..512")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=120, help="Distributed operation timeout")
    parser.add_argument("--fit-max-relative-error", type=float, default=0.35,
                        help="Export scalar settings only if every isolated pair/size median is within this fraction")
    parser.set_defaults(func=run_network)
