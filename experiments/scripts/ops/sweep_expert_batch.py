#!/usr/bin/env python3
"""Measure one StreamInfer BF16 expert forward as its token count grows."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import statistics
import sys


def batch_limit(free_bytes, hidden, intermediate, max_batch_size):
    # BF16 gate/up and down weights. Leave 20% free for allocator and backend overhead.
    weight_bytes = 6 * hidden * intermediate
    # Input, output, gate/up projection, gated activation, and an int32 expert ID.
    row_bytes = 2 * (2 * hidden + 3 * intermediate) + 4
    limit = max(0, (free_bytes * 4 // 5 - weight_bytes) // row_bytes)
    return limit if max_batch_size is None else min(limit, max_batch_size)


def batch_sizes(limit):
    size = 1
    while size < limit:
        yield size
        size *= 2
    if limit:
        yield limit


def measure(expert, batch, use_cutlass, warmup, iterations):
    import torch

    x = torch.randn(batch, expert.hidden_size, dtype=torch.bfloat16, device="cuda")
    if use_cutlass:
        layout = torch.tensor([batch], dtype=torch.int64, device=x.device)
    else:
        # All packed rows belong to expert 0, starting at aligned row 0.
        layout = torch.zeros(batch, dtype=torch.int32, device=x.device)
    for _ in range(warmup):
        expert(batch, x, layout)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        start.record()
        expert(batch, x, layout)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def plot_curve(rows, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes, times = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, times, marker="." if len(rows) > 50 else "o")
    ax.set_xscale("log", base=2)
    ax.set(yscale="log", xlabel="Tokens in one expert",
           ylabel="Median forward latency (ms)", title=title)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to time a StreamInfer expert.")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(f"StreamInfer BF16 expert kernels require SM80 or newer; found SM{major}{minor}.")

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from disagmoe.models.experts import MoEExpertsCUTLASS, MoEExpertsDeepGemmBF16
    from disagmoe.utils.logger import initialize_logger

    initialize_logger("expert-batch-sweep")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    use_cutlass = major == 8
    expert_type = MoEExpertsCUTLASS if use_cutlass else MoEExpertsDeepGemmBF16
    device = torch.cuda.get_device_name()
    free, total = torch.cuda.mem_get_info()
    limit = batch_limit(free, args.hidden, args.intermediate, args.max_batch_size)
    print(f"{device} | {expert_type.__name__} | free {free / 2**30:.2f} of {total / 2**30:.2f} GiB"
          f" | planned maximum B={limit}", flush=True)
    if limit < 1:
        raise RuntimeError("The memory budget is too small for one expert and one input row.")

    sizes = range(1, limit + 1) if args.dense else batch_sizes(limit)
    directory = args.output_dir
    csv_path = directory / "expert_latency.csv"
    png_path = directory / "expert_latency.png"
    if csv_path.exists() or png_path.exists():
        raise FileExistsError(f"Results already exist in {directory}. Choose a new --output-dir.")
    rows = []
    with torch.inference_mode():
        expert = expert_type(args.hidden, args.intermediate, num_experts=1, max_batch_size=limit).eval()
        for size in sizes:
            try:
                latency = measure(expert, size, use_cutlass, args.warmup, args.iterations)
            except torch.cuda.OutOfMemoryError:
                if not rows:
                    raise
                print(f"Stopped at B={size} (CUDA out of memory).", file=sys.stderr)
                break
            rows.append((size, latency))
            print(f"B={size:8d}  {latency:.6f} ms", flush=True)

    directory.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "time_ms", "statistic", "gpu", "backend",
                         "hidden", "intermediate", "planned_max_batch", "warmup", "iterations"])
        for size, latency in rows:
            writer.writerow([size, latency, "median", device, expert_type.__name__,
                             args.hidden, args.intermediate, limit, args.warmup, args.iterations])
    plot_curve(rows, f"{device} · {expert_type.__name__}\nH={args.hidden}, I={args.intermediate}", png_path)
    print(f"Largest measured B={rows[-1][0]}\n{csv_path}\n{png_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--max-batch-size", type=int, help="Cap the memory-based sweep at this batch size.")
    parser.add_argument("--dense", action="store_true",
                        help="Measure every integer batch size. Use e.g. --dense --max-batch-size 512 for a simulator profile.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    default_dir = Path("reports/expert-batch") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()
    if min(args.hidden, args.intermediate, args.warmup, args.iterations) < 1:
        parser.error("dimensions, warmup and iterations must be positive")
    if args.hidden % 8 or args.intermediate % 8:
        parser.error("hidden and intermediate must be multiples of 8")
    if args.max_batch_size is not None and args.max_batch_size < 1:
        parser.error("--max-batch-size must be positive")
    try:
        run(args)
    except (ImportError, RuntimeError, FileExistsError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
