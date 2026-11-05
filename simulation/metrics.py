"""Bounded, reproducible samples with exact counts, means, and maxima."""

import math
import random


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


class Samples:
    def __init__(self, capacity=100000, seed=0):
        if capacity < 1:
            raise ValueError("sample capacity must be positive")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.values = []
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0

    def add(self, value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("metric samples must be finite and nonnegative")
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        if len(self.values) < self.capacity:
            self.values.append(value)
        else:
            slot = self.rng.randrange(self.count)
            if slot < self.capacity:
                self.values[slot] = value

    def summary(self, scale=1.0):
        return {
            "count": self.count,
            "mean": self.total / self.count * scale if self.count else None,
            "max": self.maximum * scale if self.count else None,
            **{f"p{p}": (percentile(self.values, p / 100) * scale
                           if self.values else None) for p in (50, 90, 95, 99)},
            "quantile_sample_count": len(self.values),
            "quantiles_are_sampled": self.count > len(self.values),
        }
