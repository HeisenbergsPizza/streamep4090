"""Exact time-weighted observations of ready queues, without simulation events.

Occupancy columns remain layer aggregates: A_0..A_L, E_0..E_(L-1). Attention
counts are tokens; expert counts are copies. In-flight compute, communication,
and incomplete joins are not ready queues. Finalize A_L is retained in queue
means but excluded from all active-layer statistics.
Expert-wise scheduling decisions have a separate flattened candidate layout;
their scores/snapshots must not be indexed using the occupancy column labels.
"""
from __future__ import annotations

import gzip
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np


class QueueTraceRecorder:
    """Integrate piecewise-constant queues in bins anchored at ``start_ms``.

    ``change`` must be called at each enqueue/dequeue timestamp. Decisions are
    streamed separately and never used to reconstruct queue occupancy. Only
    observed time is saved: finish before the requested end leaves no padded
    bins, and a partially observed final bin uses its actual duration.
    """

    def __init__(self, path, *, num_gpus: int, num_layers: int,
                 ticks_per_millisecond: float = 10, interval_ms: float = 0.1,
                 start_ms: float = 0, end_ms: float | None = None,
                 policy: str | None = None, top_k: int | None = None,
                 configuration: dict | None = None):
        for name, value in (("num_gpus", num_gpus), ("num_layers", num_layers)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("ticks_per_millisecond", ticks_per_millisecond),
                            ("interval_ms", interval_ms)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(start_ms) or start_ms < 0:
            raise ValueError("start_ms must be finite and nonnegative")
        if end_ms is not None and (not math.isfinite(end_ms) or end_ms <= start_ms):
            raise ValueError("end_ms must be finite and greater than start_ms")
        self.num_gpus, self.num_layers = int(num_gpus), int(num_layers)
        self.ticks_per_ms = float(ticks_per_millisecond)
        self.interval_ms, self.start_ms = float(interval_ms), float(start_ms)
        self.end_ms = None if end_ms is None else float(end_ms)
        self.policy, self.top_k = policy, top_k
        self.configuration = dict(configuration or {})
        self.start_ticks = self.start_ms * self.ticks_per_ms
        self.end_ticks = math.inf if end_ms is None else self.end_ms * self.ticks_per_ms
        self.interval_ticks = self.interval_ms * self.ticks_per_ms
        if (not math.isfinite(self.start_ticks) or not math.isfinite(self.interval_ticks)
                or self.start_ticks + self.interval_ticks <= self.start_ticks
                or (end_ms is not None and not math.isfinite(self.end_ticks))):
            raise ValueError("trace window and interval exceed the clock range")
        prefix = Path(path).expanduser().resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.paths = {"npz": str(prefix) + ".npz",
                      "metadata": str(prefix) + ".metadata.json",
                      "decisions": str(prefix) + ".decisions.jsonl.gz"}
        if any(Path(filename).exists() for filename in self.paths.values()):
            raise FileExistsError(f"queue trace output already exists for prefix: {prefix}")
        self.counts = np.zeros((self.num_gpus, 2 * self.num_layers + 1), dtype=np.int64)
        self._active = np.zeros((3, self.num_gpus), dtype=np.int64)
        self._queue_integral = np.zeros_like(self.counts, dtype=np.float64)
        self._active_integral = np.zeros_like(self._active, dtype=np.float64)
        self._queue_means = []
        self._active_means = []
        self._edges_ticks = [self.start_ticks]
        self._bin_duration = 0.0
        self._bin_index = 0
        self._last_time = 0.0
        self._observed_start = None
        self._observed_end = None
        self.decision_count = 0
        self._metadata = None
        Path(self.paths["metadata"]).write_text(json.dumps({
            "format": "streaminfer.queue_trace", "version": 1,
            "status": "recording", "complete": False, "paths": self.paths,
        }, indent=2) + "\n", encoding="utf-8")
        self._decisions = gzip.open(self.paths["decisions"], "wt", encoding="utf-8")

    def _commit_bin(self, edge_ticks):
        if self._bin_duration <= 0:
            return
        self._queue_means.append(self._queue_integral / self._bin_duration)
        self._active_means.append(self._active_integral / self._bin_duration)
        self._edges_ticks.append(float(edge_ticks))
        self._queue_integral.fill(0)
        self._active_integral.fill(0)
        self._bin_duration = 0.0
        self._bin_index += 1

    def advance(self, now_ticks: float):
        """Account for the unchanged state through now, clipped to the window."""
        now = float(now_ticks)
        if self._metadata is not None:
            raise RuntimeError("queue trace is already finished")
        if not math.isfinite(now) or now < self._last_time:
            raise ValueError("queue trace time must be finite and monotonic")
        if now == self._last_time:
            return
        cursor = max(self._last_time, self.start_ticks)
        end = min(now, self.end_ticks)
        while cursor < end:
            boundary = self.start_ticks + (self._bin_index + 1) * self.interval_ticks
            segment_end = min(end, boundary)
            duration = segment_end - cursor
            if duration <= 0:
                raise ValueError("queue trace interval is below clock precision")
            self._queue_integral += self.counts * duration
            self._active_integral += self._active * duration
            self._bin_duration += duration
            if self._observed_start is None:
                self._observed_start = cursor
            self._observed_end = segment_end
            cursor = segment_end
            if cursor == boundary:
                self._commit_bin(cursor)
        self._last_time = now

    def change(self, now_ticks: float, gpu: int, column: int, delta: int):
        """Record one logical queue change; zero-time changes have zero area."""
        if not 0 <= gpu < self.num_gpus or not 0 <= column < self.counts.shape[1]:
            raise ValueError("queue trace GPU or column is out of range")
        if isinstance(delta, bool) or not isinstance(delta, Integral):
            raise ValueError("queue trace delta must be an integer")
        self.advance(now_ticks)
        if now_ticks >= self.end_ticks:
            return
        previous = int(self.counts[gpu, column])
        updated = previous + int(delta)
        if updated < 0:
            raise ValueError("ready queue count cannot become negative")
        self.counts[gpu, column] = updated
        # Only a zero/nonzero transition can change active-layer counts.
        if column != self.num_layers and (previous > 0) != (updated > 0):
            step = 1 if updated > 0 else -1
            attention = column < self.num_layers
            self._active[1 if attention else 2, gpu] += step
            other_column = (column + self.num_layers + 1 if attention
                            else column - self.num_layers - 1)
            if self.counts[gpu, other_column] == 0:
                self._active[0, gpu] += step

    def observes_schedule(self, start, end):
        """Whether a dispatch needs detailed recording; this never changes state."""
        overlaps = start < self.end_ticks and end > self.start_ticks
        instantaneous = start == end and self.start_ticks <= start < self.end_ticks
        return overlaps or instantaneous

    def capture_schedule(self, trace):
        """Save batch service intersecting the window, including zero-time decisions.

        With overheads enabled, start/end include the reserved CPU gap before
        GPU execution. gpu_start and compute_start retain the internal bounds.
        """
        start = float(trace["start"])
        end = float(trace["end"])
        if self.observes_schedule(start, end):
            record = dict(trace)
            record.update(start_ms=start / self.ticks_per_ms,
                          end_ms=end / self.ticks_per_ms,
                          time_unit="ticks", ticks_per_millisecond=self.ticks_per_ms)
            self._decisions.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            self.decision_count += 1

    def close(self):
        """Close the streaming file, including when the simulator raises."""
        self._decisions.close()

    def finish(self, now_ticks: float):
        if self._metadata is not None:
            return self._metadata
        self.advance(now_ticks)
        if self._bin_duration > 0:
            self._commit_bin(self._observed_end)
        self.close()
        bins = len(self._queue_means)
        shape = (bins, self.num_gpus, 2 * self.num_layers + 1)
        queues = np.asarray(self._queue_means, dtype=np.float64).reshape(shape)
        active = np.asarray(self._active_means, dtype=np.float64).reshape(bins, 3, self.num_gpus)
        edges = np.asarray(self._edges_ticks, dtype=np.float64) / self.ticks_per_ms
        np.savez_compressed(self.paths["npz"], queue_mean=queues, time_edges_ms=edges,
                            active_layers_mean=active[:, 0],
                            attention_active_layers_mean=active[:, 1],
                            expert_active_layers_mean=active[:, 2])
        observed = {"start": None if self._observed_start is None else self._observed_start / self.ticks_per_ms,
                    "end": None if self._observed_end is None else self._observed_end / self.ticks_per_ms}
        overhead_enabled = self.configuration.get("batch_overhead_model", {}).get("enabled", False)
        decision_window = "positive-duration scheduled compute intersects [requested start, requested end), or zero-duration decision starts inside it; carry-in intervals retained; end is scheduled completion, possibly beyond observed coverage"
        queue_definition = self.configuration.get("queue_definition", "ready, waiting work only; excludes executing work, network transfers, and incomplete joins")
        if overhead_enabled:
            decision_window = "positive-duration reserved batch service (exposed CPU gap, GPU reformation, base compute) intersects [requested start, requested end), or zero-duration decision starts inside it; carry-in intervals retained; gpu_start/compute_start delimit GPU/base compute in ticks; end is scheduled completion, possibly beyond observed coverage"
            queue_definition += "; selected batches leave the ready queue when reserved, including during exposed CPU scheduling before GPU execution"
        metadata = {
            "format": "streaminfer.queue_trace", "version": 1, "paths": self.paths,
            "status": "complete", "complete": True,
            "policy": self.policy, "top_k": self.top_k,
            "simulation_config": self.configuration, "hardware_calibrated": False,
            "shape": list(shape), "num_gpus": self.num_gpus, "num_layers": self.num_layers,
            "queue_labels": ([f"A_{i}" for i in range(self.num_layers + 1)]
                             + [f"E_{i}" for i in range(self.num_layers)]),
            "occupancy_granularity": "layer_aggregate",
            "decision_queue_layout": {
                "expert_scheduling": self.configuration.get("expert_scheduling", "layer"),
                "queue_order": ("A_0..A_L, then E_(layer,local_expert) in layer-major order"
                                if self.configuration.get("expert_scheduling") == "expert"
                                else "A_0..A_L, then E_0..E_(L-1)"),
                "num_attention_queues": self.num_layers + 1,
                "experts_per_gpu": (self.configuration["num_experts"] // self.num_gpus
                                    if "num_experts" in self.configuration else None),
                "expert_id_mapping": "gpu * experts_per_gpu + local_expert_id",
            },
            "finalize_queue_index": self.num_layers, "interval_ms": self.interval_ms,
            "ticks_per_millisecond": self.ticks_per_ms,
            "requested_window_ms": {"start": self.start_ms, "end": self.end_ms},
            "observed_window_ms": observed,
            "observed_duration_ms": 0.0 if not bins else float(edges[-1] - edges[0]),
            "simulation_end_ms": float(now_ticks) / self.ticks_per_ms,
            "decision_count": self.decision_count,
            "decision_window": decision_window,
            "queue_units": {"attention": "tokens", "expert": "expert_copies"},
            "queue_definition": queue_definition,
            "active_layer_definition": "logical: count(A_l > 0 OR E_l > 0); attention/expert: nonempty queues of that stage; all exclude finalize A_L",
            "aggregation": "exact queue-change time integral divided by actual observed bin duration; no snapshot interpolation or unobserved padding",
        }
        Path(self.paths["metadata"]).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        self._metadata = metadata
        return metadata


__all__ = ["QueueTraceRecorder"]
