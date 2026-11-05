"""Plot time-integrated scheduler queues and actual execution separately.

Run ``python -m simulation.plot_scheduler_trace --help``.  Queue colors show
bin-time means, never occupancy inferred from sampled scheduling decisions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import gzip
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QueueTrace:
    path: Path
    metadata: dict
    time_edges_ms: np.ndarray
    queue_mean: np.ndarray
    active_layers_mean: np.ndarray
    attention_active_layers_mean: np.ndarray
    expert_active_layers_mean: np.ndarray
    display_label: str | None = None

    @property
    def policy(self):
        if self.display_label is not None:
            return self.display_label
        return self.base_policy if self.expert_scheduling == "layer" else f"{self.base_policy}/expert"

    @property
    def base_policy(self):
        return self.metadata["policy"]

    @property
    def expert_scheduling(self):
        return self.metadata.get("simulation_config", {}).get("expert_scheduling", "layer")

    @property
    def layers(self):
        return self.metadata["num_layers"]

    @property
    def gpus(self):
        return self.metadata["num_gpus"]

    @property
    def top_k(self):
        return self.metadata["top_k"]


def _positive_integer(metadata, key):
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"metadata {key} must be a positive integer")
    return value


def load_trace(path: str | Path) -> QueueTrace:
    """Read and validate the recorder's versioned NPZ/metadata pair."""
    path = Path(path).resolve()
    if path.suffix != ".npz":
        raise ValueError(f"expected a .npz queue trace: {path}")
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    if metadata.get("format") != "streaminfer.queue_trace" or metadata.get("version") != 1:
        raise ValueError("unsupported queue trace format/version")
    if metadata.get("complete") is not True or metadata.get("status") != "complete":
        raise ValueError("queue trace recording is incomplete; refusing to plot partial artifacts")
    layers = _positive_integer(metadata, "num_layers")
    gpus = _positive_integer(metadata, "num_gpus")
    _positive_integer(metadata, "top_k")
    if not isinstance(metadata.get("policy"), str) or not metadata["policy"].strip():
        raise ValueError("metadata policy must be a nonempty string")
    configuration = metadata.get("simulation_config", {})
    if not isinstance(configuration, dict) or configuration.get("expert_scheduling", "layer") not in ("layer", "expert"):
        raise ValueError("metadata simulation_config.expert_scheduling must be layer or expert")
    expected_labels = [f"A_{i}" for i in range(layers + 1)] + [f"E_{i}" for i in range(layers)]
    if metadata.get("queue_labels") != expected_labels:
        raise ValueError("queue_labels must order A_0..A_L, E_0..E_(L-1)")
    if metadata.get("finalize_queue_index") != layers:
        raise ValueError("finalize_queue_index must equal num_layers")
    names = ("time_edges_ms", "queue_mean", "active_layers_mean",
             "attention_active_layers_mean", "expert_active_layers_mean")
    with np.load(path, allow_pickle=False) as archive:
        if any(name not in archive for name in names):
            raise ValueError("queue trace is missing required arrays")
        arrays = {name: np.asarray(archive[name]) for name in names}
    for name, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
            raise ValueError(f"{name} must contain real numeric values")
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    edges, queues = arrays["time_edges_ms"], arrays["queue_mean"]
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("trace needs nonempty, strictly increasing time_edges_ms")
    shape = (len(edges) - 1, gpus, 2 * layers + 1)
    if queues.shape != shape or metadata.get("shape") != list(shape):
        raise ValueError(f"queue_mean/metadata shape must be {shape}")
    for name in names[2:]:
        if arrays[name].shape != shape[:2] or np.any(arrays[name] > layers + 1e-8):
            raise ValueError(f"{name} must have shape {shape[:2]} and values <= num_layers")
    active = arrays["active_layers_mean"]
    attn = arrays["attention_active_layers_mean"]
    expert = arrays["expert_active_layers_mean"]
    if np.any(active + 1e-8 < np.maximum(attn, expert)) or np.any(active > attn + expert + 1e-8):
        raise ValueError("active layer means violate the A-or-E union bounds")
    return QueueTrace(path=path, metadata=metadata, **arrays)


def _edge_index(edges, value):
    if not math.isfinite(value):
        raise ValueError("window boundaries must be finite")
    index = int(np.searchsorted(edges, value))
    for candidate in (index, index - 1):
        if 0 <= candidate < len(edges) and np.isclose(edges[candidate], value, rtol=1e-10, atol=1e-9):
            return candidate
    raise ValueError(f"window boundary {value:g} ms is not a recorded bin edge; partial bins cannot be recut exactly")


def crop_trace(trace: QueueTrace, start_ms: float, end_ms: float) -> QueueTrace:
    """Keep complete bins only; reject unavailable or partial-bin windows."""
    first = _edge_index(trace.time_edges_ms, start_ms)
    last = _edge_index(trace.time_edges_ms, end_ms)
    if last <= first:
        raise ValueError("end_ms must exceed start_ms")
    return replace(trace, time_edges_ms=trace.time_edges_ms[first:last + 1],
                   queue_mean=trace.queue_mean[first:last],
                   active_layers_mean=trace.active_layers_mean[first:last],
                   attention_active_layers_mean=trace.attention_active_layers_mean[first:last],
                   expert_active_layers_mean=trace.expert_active_layers_mean[first:last])


def common_window(traces, start_ms=None, end_ms=None):
    """Resolve one common complete-bin window, reporting exact saved edges."""
    if not traces:
        raise ValueError("at least one trace is required")
    if len({trace.policy for trace in traces}) != len(traces):
        raise ValueError("provide distinct strategy labels; duplicate policies/labels are ambiguous")
    if len({(trace.layers, trace.gpus, trace.top_k) for trace in traces}) != 1:
        raise ValueError("policies must share num_layers, num_gpus and top_k")
    if start_ms is not None and end_ms is not None:
        start, end = start_ms, end_ms
    else:
        common = traces[0].time_edges_ms
        for trace in traces[1:]:
            edges = trace.time_edges_ms
            right = np.minimum(np.searchsorted(edges, common), len(edges) - 1)
            left = np.maximum(right - 1, 0)
            match = (np.isclose(edges[right], common, rtol=1e-10, atol=1e-9)
                     | np.isclose(edges[left], common, rtol=1e-10, atol=1e-9))
            common = common[match]
        if len(common) < 2:
            raise ValueError("traces have no common complete-bin time window")
        start = common[0] if start_ms is None else start_ms
        end = common[-1] if end_ms is None else end_ms
    cropped = [crop_trace(trace, start, end) for trace in traces]
    return cropped, float(cropped[0].time_edges_ms[0]), float(cropped[0].time_edges_ms[-1])


def window_summary(trace: QueueTrace):
    """Exact duration-weighted means over all selected complete bins, per GPU."""
    widths = np.diff(trace.time_edges_ms)
    weights = widths / widths.sum()
    queue_means = np.tensordot(weights, trace.queue_mean, axes=(0, 0))
    active = np.tensordot(weights, trace.active_layers_mean, axes=(0, 0))
    attention = np.tensordot(weights, trace.attention_active_layers_mean, axes=(0, 0))
    expert = np.tensordot(weights, trace.expert_active_layers_mean, axes=(0, 0))
    rows = []
    for gpu in range(trace.gpus):
        expert_copies = float(queue_means[gpu, trace.layers + 1:].sum())
        rows.append({"policy": trace.policy, "base_policy": trace.base_policy,
                     "expert_scheduling": trace.expert_scheduling, "gpu": gpu,
                     "start_ms": float(trace.time_edges_ms[0]), "end_ms": float(trace.time_edges_ms[-1]),
                     "duration_ms": float(widths.sum()), "bins": len(widths),
                     "active_ready_layers_time_mean": float(active[gpu]),
                     "attention_active_layers_time_mean": float(attention[gpu]),
                     "expert_active_layers_time_mean": float(expert[gpu]),
                     "attention_ready_tokens_time_mean": float(queue_means[gpu, :trace.layers].sum()),
                     "expert_ready_copies_time_mean": expert_copies,
                     "expert_ready_token_equivalents_time_mean": expert_copies / trace.top_k,
                     "finalize_ready_tokens_time_mean": float(queue_means[gpu, trace.layers]),
                     "finalize_excluded_from_active_layers": True})
    return rows


def _decision_path(trace):
    value = trace.metadata.get("paths", {}).get("decisions")
    path = Path(value) if value else trace.path.with_suffix(".decisions.jsonl.gz")
    if not path.is_absolute():
        path = trace.path.parent / path
    # Artifacts can be relocated together without retaining their original root.
    sibling = trace.path.with_suffix(".decisions.jsonl.gz")
    if not path.is_file() and sibling.is_file():
        path = sibling
    if not path.is_file():
        raise ValueError(f"decision trace is missing: {path}")
    return path


def execution_intervals(trace: QueueTrace, gpu: int):
    """Clip GPU execution (including reformation), excluding exposed CPU gaps."""
    first, last = trace.time_edges_ms[[0, -1]]
    intervals = {"attention": [], "expert": []}
    with gzip.open(_decision_path(trace), "rt", encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("gpu") != gpu or record.get("stage") == "finalize":
                continue
            stage = record.get("stage")
            if stage not in intervals:
                raise ValueError(f"unknown decision stage on line {number}: {stage}")
            start, end = record.get("start_ms"), record.get("end_ms")
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                    or not math.isfinite(start) or not math.isfinite(end) or end < start):
                raise ValueError(f"invalid decision time on line {number}")
            if "gpu_start" in record:
                # Overhead-enabled start/end cover reserved batch service;
                # gpu_start is in ticks, not milliseconds. Old traces retain
                # their original start_ms semantics without this field.
                gpu_start = record["gpu_start"]
                ticks_per_ms = record.get("ticks_per_millisecond",
                                          trace.metadata.get("ticks_per_millisecond"))
                if (isinstance(gpu_start, bool) or isinstance(ticks_per_ms, bool)
                        or not isinstance(gpu_start, (int, float))
                        or not isinstance(ticks_per_ms, (int, float))
                        or not math.isfinite(gpu_start) or not math.isfinite(ticks_per_ms)
                        or ticks_per_ms <= 0):
                    raise ValueError(f"invalid GPU start/unit on line {number}")
                gpu_start_ms = gpu_start / ticks_per_ms
                if not start <= gpu_start_ms <= end:
                    raise ValueError(f"GPU interval outside batch service on line {number}")
                start = gpu_start_ms
            layer = record.get("layer")
            if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < trace.layers:
                raise ValueError(f"invalid decision layer on line {number}")
            lo, hi = max(float(first), start), min(float(last), end)
            if hi > lo:
                intervals[stage].append((lo, hi, layer))
    return intervals


def _write_csv(path, rows):
    rows = iter(rows)
    first = next(rows)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def _active_rows(traces):
    for trace in traces:
        for index, (start, end) in enumerate(zip(trace.time_edges_ms[:-1], trace.time_edges_ms[1:])):
            for gpu in range(trace.gpus):
                yield {"policy": trace.policy, "base_policy": trace.base_policy,
                       "expert_scheduling": trace.expert_scheduling, "gpu": gpu, "bin_start_ms": float(start),
                       "bin_end_ms": float(end), "duration_ms": float(end - start),
                       "active_layers_time_mean": float(trace.active_layers_mean[index, gpu]),
                       "attention_active_layers_time_mean": float(trace.attention_active_layers_mean[index, gpu]),
                       "expert_active_layers_time_mean": float(trace.expert_active_layers_mean[index, gpu])}


def _layer_rows(traces):
    for trace in traces:
        means = np.average(trace.queue_mean, axis=0, weights=np.diff(trace.time_edges_ms))
        for gpu in range(trace.gpus):
            for stage, offset in (("attention", 0), ("expert", trace.layers + 1)):
                for layer in range(trace.layers):
                    raw = float(means[gpu, offset + layer])
                    yield {"policy": trace.policy, "base_policy": trace.base_policy,
                           "expert_scheduling": trace.expert_scheduling, "gpu": gpu, "stage": stage, "layer": layer,
                           "start_ms": float(trace.time_edges_ms[0]), "end_ms": float(trace.time_edges_ms[-1]),
                           "ready_time_mean_raw": raw,
                           "raw_unit": "tokens" if stage == "attention" else "expert_copies",
                           "ready_time_mean_token_equivalents": raw if stage == "attention" else raw / trace.top_k}


def render(traces, output_dir, gpu=0, title="Scheduler queues", expert_scale="normalized", dpi=180,
           requested_start_ms=None, requested_end_ms=None):
    """Create figures and exact-window CSVs. All input GPUs remain in the CSVs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    if any(not 0 <= gpu < trace.gpus for trace in traces):
        raise ValueError(f"GPU {gpu} is outside the recorded range")
    if expert_scale not in ("normalized", "raw"):
        raise ValueError("expert_scale must be normalized or raw")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"output directory must be absent or empty: {output_dir}")
    executions = [execution_intervals(trace, gpu) for trace in traces]
    output_dir.mkdir(parents=True, exist_ok=True)
    layers, count = traces[0].layers, len(traces)
    start, end = map(float, traces[0].time_edges_ms[[0, -1]])
    heading = title if "simulation proxy" in title.lower() else f"Simulation proxy · {title}"
    window_text = f"GPU {gpu} · {start:g}–{end:g} ms · finalize A_{layers} excluded"
    matrices = [(trace.queue_mean[:, gpu, :layers].T,
                 trace.queue_mean[:, gpu, layers + 1:].T / (trace.top_k if expert_scale == "normalized" else 1))
                for trace in traces]
    # Each stage's scale is fixed across all policies. Raw E counts are expert
    # copies, so they intentionally do not share a numeric scale with A tokens.
    maxima = [max(float(pair[stage].max()) for pair in matrices) for stage in (0, 1)]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    style = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
             "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": dpi}
    files = []

    def save(figure, stem):
        for extension in ("png", "pdf"):
            path = output_dir / f"{stem}.{extension}"
            figure.savefig(path, bbox_inches="tight", facecolor="white")
            files.append(path.name)
        plt.close(figure)

    with plt.rc_context(style):
        figure, axes = plt.subplots(count, 2, figsize=(12.4, 1.9 * count + 1.4), squeeze=False,
                                   sharex=True, sharey=True, layout="constrained")
        meshes = [None, None]
        for row, (trace, pair) in enumerate(zip(traces, matrices)):
            for column, values in enumerate(pair):
                axis = axes[row, column]
                meshes[column] = axis.pcolormesh(trace.time_edges_ms, np.arange(layers + 1) - .5,
                                                 np.ma.masked_where(values == 0, np.log1p(values)),
                                                 cmap=cmap, shading="flat",
                                                 norm=Normalize(0, max(np.log1p(maxima[column]), 1e-9)),
                                                 rasterized=True)
                axis.set_ylim(layers - .5, -.5)
                axis.set_xlim(start, end)
                axis.set_yticks(np.unique(np.linspace(0, layers - 1, min(layers, 6)).astype(int)))
                if column == 0:
                    axis.set_ylabel(f"{trace.policy}\nModel layer")
                if row == 0:
                    axis.set_title("Attention ready tokens" if column == 0 else
                                   ("Expert ready copies / Top-k" if expert_scale == "normalized" else "Expert ready copies"))
                if row == count - 1:
                    axis.set_xlabel("Simulation time (ms)")
        for column in (0, 1):
            colorbar = figure.colorbar(meshes[column], ax=axes[:, column].tolist(), location="bottom", shrink=.88, pad=.04)
            maximum = maxima[column]
            ticks = np.expm1(np.linspace(0, np.log1p(maximum), 5)) if maximum else np.array([0.0])
            colorbar.set_ticks(np.log1p(ticks), labels=[f"{tick:.3g}" for tick in ticks])
            colorbar.set_label("Time-mean ready depth (log1p color scale; white = 0)")
        figure.suptitle(f"{heading}\n{window_text}", fontsize=12)
        save(figure, "scheduler_heatmap")

        figure, axes = plt.subplots(count, 1, figsize=(12.4, 1.7 * count + 1.3), squeeze=False,
                                   sharex=True, sharey=True, layout="constrained")
        for row, (trace, intervals) in enumerate(zip(traces, executions)):
            axis = axes[row, 0]
            for stage, color in (("attention", "#2878b5"), ("expert", "#df7c25")):
                polygons = [[(lo, layer - .4), (hi, layer - .4), (hi, layer + .4), (lo, layer + .4)]
                            for lo, hi, layer in intervals[stage]]
                axis.add_collection(PolyCollection(polygons, facecolors=color, edgecolors="none", rasterized=True))
            axis.set(xlim=(start, end), ylim=(layers - .5, -.5), ylabel=f"{trace.policy}\nModel layer")
            axis.set_yticks(np.unique(np.linspace(0, layers - 1, min(layers, 6)).astype(int)))
        axes[-1, 0].set_xlabel("Simulation time (ms)")
        axes[0, 0].legend(handles=[Patch(color="#2878b5", label="Attention execution"),
                                  Patch(color="#df7c25", label="Expert execution (group or single)")],
                         loc="lower right", bbox_to_anchor=(1, 1.01), borderaxespad=0,
                         ncol=2, fontsize=8)
        figure.suptitle(f"{heading} · Actual execution intervals\n{window_text}\n"
                       "Layer coordinates; individual expert_id is retained in decision traces", fontsize=12)
        save(figure, "scheduler_execution")

        figure, axis = plt.subplots(figsize=(12.4, 3.8), layout="constrained")
        for trace in traces:
            axis.stairs(trace.active_layers_mean[:, gpu], trace.time_edges_ms, label=trace.policy, linewidth=1.25)
        axis.set(xlim=(start, end), ylim=(0, layers), xlabel="Simulation time (ms)",
                 ylabel="Active ready layers\n(time mean within each bin)")
        axis.grid(axis="y", color=".9", linewidth=.6)
        axis.legend(ncol=min(count, 4), loc="upper right")
        axis.set_title(f"{heading}\n{window_text}; A_l > 0 OR E_l > 0", fontsize=12)
        save(figure, "active_layers")

    _write_csv(output_dir / "window_summary.csv", (row for trace in traces for row in window_summary(trace)))
    _write_csv(output_dir / "active_layers.csv", _active_rows(traces))
    _write_csv(output_dir / "layer_ready_summary.csv", _layer_rows(traces))
    manifest = {"format": "streaminfer.scheduler_trace_plot", "version": 1,
                "title": heading, "gpu": gpu, "expert_scale": expert_scale,
                "requested_window_ms": {"start": requested_start_ms, "end": requested_end_ms},
                "actual_window_ms": {"start": start, "end": end},
                "window_rule": "common complete recorded bins; partial-bin user boundaries rejected",
                "finalize_excluded_from_figures_and_active_layers": True,
                "summary_semantics": "duration-weighted integrals over the selected window; all GPUs retained in CSV",
                "active_layers_semantics": "time-integral of count(A_l > 0 OR E_l > 0), never count of positive bin means",
                "execution_semantics": "recorded service intervals clipped to the window; finalize excluded; plotted at layer coordinates, not individual expert rows",
                "execution_coordinates": "layer",
                "expert_identity_source": "expert_id in each expert-wise decision trace record; same-layer experts share a plot row",
                "color_scale": "log1p time-mean ready depth; one shared scale per stage across policies",
                "color_maxima_before_log1p": {"attention": maxima[0], "expert": maxima[1]},
                "traces": [{"policy": trace.policy, "base_policy": trace.base_policy,
                            "expert_scheduling": trace.expert_scheduling, "npz": str(trace.path),
                            "metadata": str(trace.path.with_suffix('.metadata.json')),
                            "decisions": str(_decision_path(trace)),
                            "decision_window": trace.metadata.get("decision_window"),
                            "bins": len(trace.time_edges_ms) - 1} for trace in traces],
                "figures": files}
    (output_dir / "plot_metadata.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", help="optional display labels in the same order as --traces")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start-ms", type=float)
    parser.add_argument("--end-ms", type=float)
    parser.add_argument("--expert-scale", choices=("normalized", "raw"), default="normalized")
    parser.add_argument("--title", default="Scheduler queue comparison")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    try:
        if args.dpi < 1:
            raise ValueError("dpi must be positive")
        traces = [load_trace(path) for path in args.traces]
        if args.labels is not None:
            if len(args.labels) != len(traces) or any(not label.strip() for label in args.labels):
                raise ValueError("--labels requires one nonempty label per trace")
            traces = [replace(trace, display_label=label) for trace, label in zip(traces, args.labels)]
        order = {policy: index for index, policy in enumerate(("defrag", "flfs", "mtfs", "fifo"))}
        traces.sort(key=lambda trace: (order.get(trace.base_policy, len(order)), trace.policy))
        traces, start, end = common_window(traces, args.start_ms, args.end_ms)
        render(traces, args.output_dir, gpu=args.gpu, title=args.title,
               expert_scale=args.expert_scale, dpi=args.dpi,
               requested_start_ms=args.start_ms, requested_end_ms=args.end_ms)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output_dir.resolve()}; actual window [{start:g}, {end:g}) ms; GPU {args.gpu}")


if __name__ == "__main__":
    main()
