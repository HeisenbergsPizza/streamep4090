"""Plot successful StreamInfer dequeues and driver-observed outstanding requests."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


STAGES = {
    "attention": ("Attention", "#2166ac", "o"),
    "expert": ("Expert", "#d86916", "^"),
    "finalize": ("Finalize", "#535963", "s"),
}


def load_run(directory):
    run = json.loads((directory / "run.json").read_text())
    if run["format"] != "streaminfer.scheduler_choices.v1" or run["generator_type"] != "uniform":
        raise ValueError("Expected a fixed-rate StreamInfer scheduler choices run")
    start, end = run["start_ns"], run["end_ns"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("Invalid measurement interval in run.json")
    if type(run["num_layers"]) is not int or run["num_layers"] < 1:
        raise ValueError("Invalid model layer count in run.json")
    devices = run["device_ids"]
    if (not isinstance(devices, list) or not devices
            or any(type(d) is not int or d < 0 for d in devices) or len(set(devices)) != len(devices)):
        raise ValueError("Invalid device IDs in run.json")
    if not math.isfinite(run["rate"]) or run["rate"] <= 0:
        raise ValueError("Invalid request rate in run.json")
    submits, finishes = run["submitted_s"], run["finished_s"]
    duration_s = (end - start) / 1e9
    if not submits or len(submits) != len(finishes):
        raise ValueError("Expected one completion for each submitted request")
    if any(not math.isfinite(t) or not 0 <= t <= duration_s + 1e-9 for t in (*submits, *finishes)):
        raise ValueError("Request timestamps outside measurement interval")
    return run


def outstanding_steps(run):
    changes = Counter(run["submitted_s"])
    changes.subtract(run["finished_s"])
    times, counts = [0.], [0]
    for t, delta in sorted(changes.items()):
        times.append(t)
        counts.append(counts[-1] + delta)
        if counts[-1] < 0:
            raise ValueError("More completions than submissions at this point in the run")
    return times, counts


def selected_window(run, start_s, duration_ms):
    if not math.isfinite(start_s) or not math.isfinite(duration_ms) or start_s < 0 or duration_ms <= 0:
        raise ValueError("Require finite --start-s >= 0 and --duration-ms > 0")
    start = run["start_ns"] + round(start_s * 1e9)
    end = start + round(duration_ms * 1e6)
    if end <= start or end > run["end_ns"]:
        raise ValueError("Requested window is empty or exceeds the recorded run")
    return start, end


def read_choices(directory, run, gpu, start, end):
    path = directory / f"choices-gpu-{gpu}.csv"
    choices = {kind: [] for kind in STAGES}
    previous = run["start_ns"]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"time_ns", "stage", "layer", "batch_size"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing scheduler choice CSV columns in {path}")
        for row in reader:
            t, stage = int(row["time_ns"]), row["stage"]
            layer, size = int(row["layer"]), int(row["batch_size"])
            if not previous <= t < run["end_ns"]:
                raise ValueError(f"{path}:{reader.line_num}: unordered or out-of-run timestamp")
            previous = t
            if (stage not in STAGES or size < 1 or not 0 <= layer <= run["num_layers"]
                    or (stage == "finalize") != (layer == run["num_layers"])):
                raise ValueError(f"{path}:{reader.line_num}: invalid stage, layer or batch size")
            if start <= t < end:
                choices[stage].append(((t - start) / 1e6, layer))
    return choices


def plot(directory, output, start_s, duration_ms, gpu=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    run = load_run(directory)
    start, end = selected_window(run, start_s, duration_ms)
    devices = sorted(run["device_ids"]) if gpu is None else [gpu]
    if gpu is not None and gpu not in run["device_ids"]:
        raise ValueError(f"GPU {gpu} is outside the recorded devices {run['device_ids']}")
    rows = [read_choices(directory, run, d, start, end) for d in devices]
    times, counts = outstanding_steps(run)
    lo, hi = ((t - run["start_ns"]) / 1e9 for t in (start, end))
    last_submit = max(run["submitted_s"])
    duration = (run["end_ns"] - run["start_ns"]) / 1e9
    layers = run["num_layers"]
    size = (12, 2.8 + 1.55 * len(devices))
    fig, (overview, *axes) = plt.subplots(len(devices) + 1, 1, figsize=size, layout="constrained")
    try:
        overview.step([*times, duration], [*counts, counts[-1]], where="post", color="#173a54", lw=1.2)
        overview.axvspan(last_submit, duration, color="#c7cbd0", alpha=.45, label="After last submission")
        overview.axvspan(lo, hi, color="#9470be", alpha=.25)
        overview.axvline(lo, color="#714a97", lw=1, label="Selected window")
        overview.legend(fontsize=8, loc="upper left", ncol=2)
        overview.set(xlim=(0, duration), ylabel="Outstanding requests", xlabel="Time from benchmark start (s)")
        overview.set_title(f"Fixed arrivals at {run['rate']:g} requests/s   ·   {len(run['submitted_s'])} requests",
                           loc="left", fontsize=12)
        overview.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        finalize_y = layers + max(1, layers // 15)
        ticks = list(range(0, layers, max(1, math.ceil(layers / 4))))
        for axis, device, choices in zip(axes, devices, rows):
            count = sum(map(len, choices.values()))
            for stage, values in choices.items():
                if values:
                    x, y = zip(*values)
                    if stage == "finalize":
                        y = [finalize_y] * len(y)
                    _, color, marker = STAGES[stage]
                    axis.scatter(x, y, s=13, marker=marker, c=color, linewidths=0)
            # Keep layer 0 at the top, matching simulation/plot_scheduler_trace.py.
            axis.set(xlim=(0, (end - start) / 1e6), ylim=(finalize_y + 2, -2),
                     yticks=[*ticks, finalize_y], yticklabels=[*map(str, ticks), "F"],
                     ylabel=f"GPU {device}\nLayer")
            axis.grid(alpha=.15)
            axis.set_axisbelow(True)
            axis.text(.995, .02, f"{count} selections", ha="right", va="bottom", fontsize=8,
                      transform=axis.transAxes, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        handles = [Line2D([], [], marker=m, color=c, ls="none", label=label, markersize=5)
                   for label, c, m in STAGES.values()]
        axes[0].legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1), ncol=3, fontsize=8)
        axes[0].set_title("Successful scheduler dequeues", loc="left", fontsize=11)
        axes[-1].set_xlabel(f"Time after {lo:g} s (ms)")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=170)
    finally:
        plt.close(fig)
    if hi > last_submit:
        print(f"Window includes drain after the last submission at {last_submit:g} s.")
    print(f"Wrote {output} for [{lo:g}, {hi:g}) s; "
          + ", ".join(f"GPU {d}: {sum(map(len, r.values()))}" for d, r in zip(devices, rows)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start-s", type=float, required=True, help="Seconds from measured workload start")
    parser.add_argument("--duration-ms", type=float, required=True)
    parser.add_argument("--gpu", type=int, help="Limit to a single GPU; default is all recorded GPUs")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG file")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be a .png file")
    try:
        plot(args.run_dir, args.output, args.start_s, args.duration_ms, args.gpu)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
