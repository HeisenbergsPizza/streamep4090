"""Time StreamInfer's eager gather kernels, independent of CPU and pointer uploads."""

import csv
import ctypes
from itertools import combinations
from pathlib import Path
import random
from statistics import median

from .common import StageWriter, load_manifest, write_json


def points(output_rows, fragments, topk):
    """B counts destination rows, N counts source rows and F counts source spans."""
    for rows in sorted(set(output_rows)):
        for name, factor in (("gather", 1), ("gather_and_sum", topk)):
            # The engine uses ordinary merge for Top-1.
            if name == "gather_and_sum" and factor == 1:
                continue
            inputs = rows * factor
            for count in sorted(set(fragments)):
                if count <= inputs:
                    yield dict(kind="bypass" if name == "gather" and count == 1 else name,
                               output_rows=rows, input_rows=inputs, fragments=count, top_k=factor)


def source_indices(rows, fragments, gap_rows=1, order="forward"):
    """Contiguous spans separated by an explicit physical gap in a backing tensor."""
    sizes = [rows // fragments + (i < rows % fragments) for i in range(fragments)]
    spans = []
    cursor = 0
    for size in sizes:
        spans.append(range(cursor, cursor + size))
        cursor += size + gap_rows
    if order == "shuffled":
        random.Random(1919).shuffle(spans)
    return [idx for span in spans for idx in span]


def capture(point, width, device, warmup, repeats, gap_rows, order):
    import torch

    if point["kind"] == "bypass":
        # TokenBatch::merge returns its only contiguous input without a launch.
        return []
    n = point["input_rows"]
    b = point["output_rows"]
    k = point["top_k"]
    indices = source_indices(n, point["fragments"], gap_rows, order)
    source = torch.empty((max(indices) + 1, width), dtype=torch.bfloat16, device=device)
    values = (torch.arange(source.size(0), device=device) % 17 - 8).to(source.dtype)
    source.copy_(values[:, None])
    out = torch.empty((b, width), dtype=source.dtype, device=device)
    stride = source.stride(0) * source.element_size()
    pointers = (ctypes.c_uint64 * n)(*(source.data_ptr() + idx * stride for idx in indices))
    args = (out, ctypes.addressof(pointers), b)
    if point["kind"] == "gather":
        operation = torch.ops.disag_ops.gather_tokens
    else:
        operation = torch.ops.disag_ops.gather_and_sum_tokens
        args += (k,)
    args += (width,)
    for _ in range(warmup):
        operation(*args)
        torch.cuda.synchronize(device)
    expected = values[indices].float().view(b, k).sum(1).to(out.dtype)
    torch.testing.assert_close(out, expected[:, None].expand_as(out), rtol=0, atol=0)
    # The op internally stages row pointers (H2D on the no-GDR build). A pair
    # of CUDA events wrapping the Python call would include that and host gaps.
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise ValueError("CUPTI CUDA activity tracing is required to isolate the gather kernel.")
    with torch.profiler.profile(activities=activities) as profile:
        for _ in range(repeats):
            operation(*args)
            torch.cuda.synchronize(device)
    kernel = point["kind"] + "_tokens_kernel"
    samples = [event.time_range.elapsed_us() for event in profile.events()
               if event.device_type == torch.autograd.DeviceType.CUDA
               and event.device_index == device.index and kernel in event.name]
    if len(samples) != repeats or any(sample <= 0 for sample in samples):
        raise ValueError(f"Expected {repeats} positive {kernel} GPU durations, obtained {len(samples)}. "
                         "Check the extension, CUPTI and access to CUDA activity tracing.")
    return samples


def fit(points_measured):
    """Fit the simulator's two per-kind intercepts/slopes and one shared F term."""
    import numpy as np

    measured = [p for p in points_measured if p["kind"] != "bypass"]
    kinds = sorted({p["kind"] for p in measured})
    if not kinds:
        return {}, {"status": "no measured kernels"}
    design, observations = [], []
    for p in measured:
        # MB times (us/MB); convert the fitted slope to decimal GB/s below.
        row = []
        for kind in kinds:
            row.extend((1., p["traffic_bytes"] / 1e6) if p["kind"] == kind else (0., 0.))
        design.append([*row, p["fragments"]])
        observations.append(p["median_kernel_us"])
    matrix = np.array(design, dtype=float)
    y = np.array(observations)
    dim = matrix.shape[1]
    if np.linalg.matrix_rank(matrix) < dim or len(y) <= dim:
        return {}, {"status": "insufficient independent rows, byte sizes or fragment counts for fit"}
    # Small (3 or 5 variable) nonnegative least squares problem. Enumerating
    # active columns also permits exact zero intercepts and fragment penalties.
    best, error = np.zeros(dim), float(y @ y)
    for size in range(1, dim + 1):
        for cols in combinations(range(dim), size):
            values = np.linalg.lstsq(matrix[:, cols], y, rcond=None)[0]
            if np.any(values < 0):
                continue
            residual = y - matrix[:, cols] @ values
            squared = float(residual @ residual)
            if squared < error:
                best = np.zeros(dim)
                best[list(cols)] = values
                error = squared
    if any(best[2*i+1] <= 0 for i in range(len(kinds))):
        return {}, {"status": "timing samples do not identify a positive byte slope"}
    params = {"--reformation-per-fragment-us": float(best[-1])}
    for i, kind in enumerate(kinds):
        key = "reformation" if kind == "gather" else "reduction"
        params[f"--{key}-fixed-us"] = float(best[2*i])
        params[f"--{key}-bandwidth-gb-s"] = float(1000 / best[2*i+1])
    return params, {"status": "fit", "observations": len(y), "rmse_us": float((error / len(y))**0.5),
                    "max_abs_error_us": float(max(abs(matrix @ best - y)))}


def run(args):
    import torch
    import disagmoe_c  # registers torch.ops.disag_ops

    manifest = load_manifest(args.bundle)
    if min(*args.rows, *args.fragments, args.warmup, args.repeats, args.gap_rows) < 1:
        raise ValueError("Rows, fragments, gap rows, warmup and repeats must be positive integers.")
    if manifest["runtime"]["dtype"] != "bfloat16":
        raise ValueError("StreamInfer gather kernels currently require BF16.")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    width = manifest["model"]["hidden_size"]
    topk = manifest["model"]["top_k"]
    planned = list(points(args.rows, args.fragments, topk))
    if not any(p["kind"] != "bypass" for p in planned):
        raise ValueError("Chosen grid contains no active gather or reduction kernel.")
    with StageWriter(args.bundle, "reformation") as output:
        records = []
        with (output / "raw.csv").open("w", newline="") as f:
            fields = ["kind", "output_rows", "input_rows", "fragments", "top_k", "traffic_bytes"]
            writer = csv.DictWriter(f, fieldnames=fields + ["repeat", "kernel_us"])
            writer.writeheader()
            for point in planned:
                samples = capture(point, width, device, args.warmup, args.repeats, args.gap_rows, args.span_order)
                point["traffic_bytes"] = 0 if not samples else (point["input_rows"] + point["output_rows"]) * width * 2
                for i, value in enumerate(samples):
                    writer.writerow({**point, "repeat": i, "kernel_us": value})
                records.append({**point, "median_kernel_us": median(samples) if samples else 0.,
                                "samples": len(samples)})
        params, diagnostics = fit(records)
        write_json(output / "parameters.json", params)
        write_json(output / "details.json", dict(
            operation="native eager BF16 gather; no graph capture even when compute uses graph",
            timing="CUPTI device kernel duration only; excludes host work, pointer uploads, allocation and waits",
            layout="synthetic disjoint contiguous source spans in one backing tensor; repeated warm input",
            gap_rows=args.gap_rows, span_order=args.span_order,
            bypass="one compatible source has zero GPU reformation by construction; no timing sample",
            device=args.device, warmup=args.warmup, repeats=args.repeats,
            fit=diagnostics, grid=records))


def add_commands(commands):
    p = commands.add_parser("reformation", help="Profile StreamInfer's eager gather and Top-K reduction kernels")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256, 512],
                   help="Destination rows for both gather and reduction")
    p.add_argument("--fragments", type=int, nargs="+", default=[1, 2, 8, 32],
                   help="Contiguous source spans, excluding counts larger than source rows")
    p.add_argument("--span-order", choices=["forward", "shuffled"], default="shuffled",
                   help="Order of contiguous source spans in the destination pointer table")
    p.add_argument("--gap-rows", type=int, default=1, help="Physical gap in hidden rows between spans")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.set_defaults(func=run)
