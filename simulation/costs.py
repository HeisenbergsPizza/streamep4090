"""Explicit, shared cost contracts for all simulation execution modes.

Lookup values passed to ExpertCostModel are already in simulation ticks.
Optional CSV measurements are milliseconds and converted exactly once. These
classes never infer grouped efficiency or extrapolate missing measurements.
"""

import csv
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from numbers import Integral


def _positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def _counts(values, label):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of nonnegative integers")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence of nonnegative integers") from exc
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in result):
        raise ValueError(f"{label} must contain nonnegative integers")
    return tuple(int(value) for value in result)


def _profile_rows(path, required):
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        headers = [name.strip() for name in reader.fieldnames or []]
        if len(set(headers)) != len(headers) or not set(required).issubset(headers):
            raise ValueError(f"Profile {path} requires unique columns {', '.join(required)}")
        reader.fieldnames = headers
        rows = list(reader)
    if not rows:
        raise ValueError(f"Profile {path} has no measurements")
    return rows


class ExpertCostModel:
    def __init__(self, lookup, scale=1.0, grouped_profile_path=None,
                 ticks_per_millisecond=10):
        self.scale = _positive(scale, "expert cost scale")
        self.ticks_per_millisecond = _positive(ticks_per_millisecond, "ticks_per_millisecond")
        self.lookup = {}
        for batch_size, time in lookup.items():
            batch_size = _counts([batch_size], "lookup batch size")[0]
            if batch_size == 0:
                raise ValueError("lookup batch sizes must be positive")
            self.lookup[batch_size] = _positive(time, "lookup time")
        self.grouped_profile_path = grouped_profile_path
        self.grouped_lookup = None
        if grouped_profile_path is not None:
            self.grouped_lookup = {}
            for row in _profile_rows(grouped_profile_path, ("expert_counts", "time_ms")):
                try:
                    decoded = json.loads(row["expert_counts"])
                    if not isinstance(decoded, list):
                        raise ValueError("expert_counts must be a JSON array")
                    counts = _counts(decoded, "expert_counts")
                    if not counts:
                        raise ValueError("expert_counts must not be empty")
                    time = _positive(float(row["time_ms"]) * self.ticks_per_millisecond, "grouped time")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid grouped expert profile row: {row}") from exc
                if counts in self.grouped_lookup:
                    raise ValueError(f"Duplicate expert_counts {counts} in grouped profile")
                self.grouped_lookup[counts] = time

    def __call__(self, loads):
        counts = _counts(loads, "expert loads")
        if not any(counts):
            return 0.0
        if self.grouped_lookup is not None:
            if counts not in self.grouped_lookup:
                raise ValueError(f"Grouped expert profile has no exact measurement for expert_counts={counts}")
            result = self.grouped_lookup[counts]
        else:
            missing = sorted({count for count in counts if count and count not in self.lookup})
            if missing:
                raise ValueError(f"Expert profile missing batch sizes {missing}")
            result = sum(self.lookup[count] for count in counts if count)
        return _positive(result * self.scale, "scaled expert cost")


@dataclass(frozen=True)
class AttentionCostEstimate:
    """One pure-decode batch's planned costs; all durations are simulation ticks."""

    batch_size: int
    context_mean: float | None
    components_ticks: dict[str, float]

    @property
    def total_ticks(self):
        return math.fsum(self.components_ticks.values())


def _bracket(axis, query, label):
    if query < axis[0] or query > axis[-1]:
        raise ValueError(f"Attention profile {label}={query:g} outside measured range [{axis[0]}, {axis[-1]}]")
    index = bisect_left(axis, query)
    if axis[index] == query:
        return ((axis[index], 1.0),)
    lower, upper = axis[index - 1], axis[index]
    weight = (query - lower) / (upper - lower)
    return ((lower, 1.0 - weight), (upper, weight))


class AttentionCostModel:
    """Pure decode, q=1: interpolate measured costs at (batch, mean KV history).

    CSV: batch_size,context_length,time_ms, with optional component column.
    Without component, each row is the total attention-block cost. With it,
    each coordinate must contain the same disjoint components. Missing grid
    corners and queries outside the measured axes fail; no extrapolation,
    padding or skew correction is inferred.
    """

    def __init__(self, per_token_time, profile_path=None, ticks_per_millisecond=10):
        self.per_token_time = float(per_token_time)
        if not math.isfinite(self.per_token_time) or self.per_token_time < 0:
            raise ValueError("per_token_time must be finite and nonnegative")
        self.ticks_per_millisecond = _positive(ticks_per_millisecond, "ticks_per_millisecond")
        self.profile_path = profile_path
        self.lookup = None
        self.component_lookup = None
        self.batch_sizes = ()
        self.context_lengths = ()
        self.components = ("linear_per_token_proxy",)
        if profile_path is not None:
            self.component_lookup = {}
            for row in _profile_rows(profile_path, ("batch_size", "context_length", "time_ms")):
                try:
                    batch_size = int(row["batch_size"])
                    context_length = int(row["context_length"])
                    if batch_size <= 0 or context_length < 0:
                        raise ValueError("batch_size must be positive and context_length nonnegative")
                    component = (row["component"] or "").strip() if "component" in row else "total"
                    if not component:
                        raise ValueError("component must not be empty")
                    time = float(row["time_ms"]) * self.ticks_per_millisecond
                    if not math.isfinite(time) or time < 0:
                        raise ValueError("attention time must be finite and nonnegative")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid attention profile row: {row}") from exc
                key = (batch_size, context_length)
                values = self.component_lookup.setdefault(key, {})
                if component in values:
                    raise ValueError(f"Duplicate batch/context/component {key}/{component} in attention profile")
                values[component] = time
            self.components = tuple(sorted({name for values in self.component_lookup.values() for name in values}))
            if "total" in self.components and len(self.components) != 1:
                raise ValueError("Attention profile cannot mix total with component measurements")
            self.lookup = {}
            for key, values in self.component_lookup.items():
                if set(values) != set(self.components):
                    raise ValueError(f"Attention profile missing components at batch/context {key}")
                self.lookup[key] = _positive(math.fsum(values.values()), "attention total")
            self.batch_sizes = tuple(sorted({key[0] for key in self.lookup}))
            self.context_lengths = tuple(sorted({key[1] for key in self.lookup}))

    def evaluate(self, context_lengths):
        contexts = _counts(context_lengths, "context_lengths")
        if not contexts:
            return AttentionCostEstimate(0, None, {name: 0.0 for name in self.components})
        batch_size = len(contexts)
        context_mean = sum(contexts) / batch_size
        if self.lookup is None:
            result = batch_size * self.per_token_time
            if not math.isfinite(result):
                raise ValueError("attention cost is not finite")
            return AttentionCostEstimate(batch_size, context_mean, {self.components[0]: result})
        batches = _bracket(self.batch_sizes, batch_size, "batch_size")
        histories = _bracket(self.context_lengths, context_mean, "mean context_length")
        corners = [(b, c, wb * wc) for b, wb in batches for c, wc in histories]
        missing = [(b, c) for b, c, _ in corners if (b, c) not in self.component_lookup]
        if missing:
            raise ValueError(f"Attention profile missing interpolation corners {missing}")
        components = {
            name: math.fsum(weight * self.component_lookup[(b, c)][name]
                            for b, c, weight in corners)
            for name in self.components
        }
        return AttentionCostEstimate(batch_size, context_mean, components)

    def __call__(self, context_lengths):
        return self.evaluate(context_lengths).total_ticks

    def describe(self):
        return {
            "kind": "decode_batch_context_profile" if self.lookup is not None else "linear_per_token_proxy",
            "query_tokens_per_request": 1,
            "context_aggregation": "arithmetic_mean" if self.lookup is not None else "unused",
            "interpolation": "bilinear" if self.lookup is not None else "none",
            "out_of_range": "error" if self.lookup is not None else "not_applicable",
            "components": list(self.components),
            "batch_sizes": list(self.batch_sizes),
            "context_lengths": list(self.context_lengths),
            "ticks_per_millisecond": self.ticks_per_millisecond,
        }


__all__ = ["ExpertCostModel", "AttentionCostModel", "AttentionCostEstimate"]
