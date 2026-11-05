"""CPU-only routing trace replay for the discrete-event simulators.

Simulation request/token IDs are ordinal positions. They cycle through sorted
profile request IDs and, within each request, sorted recorded token indices.
Recorded expert IDs are preserved; no expert projection is inferred from the
subset of experts observed in a trace.
"""

from numbers import Integral
import re
from typing import Optional, Tuple

import torch


def _nonnegative_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}")
    return int(value)


class ProfileDrivenRouter:
    def __init__(self, profile_bytes: bytes, num_experts_expected: int,
                 top_k: int, layer_id: Optional[int] = None) -> None:
        self.num_experts = _nonnegative_int(num_experts_expected, "num_experts_expected")
        self.top_k = _nonnegative_int(top_k, "top_k")
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("Require 0 < top_k <= num_experts_expected")
        self.layer_id = None if layer_id is None else _nonnegative_int(layer_id, "layer_id")
        if not profile_bytes:
            raise ValueError("ProfileDrivenRouter requires non-empty profile bytes")

        # Keep importing the simulation cost models independent of Parquet support.
        import pyarrow as pa
        import pyarrow.parquet as pq

        try:
            table = pq.read_table(pa.BufferReader(profile_bytes))
        except Exception as exc:
            raise ValueError(f"Failed to parse Parquet routing profile: {exc}") from exc
        names = table.column_names
        if len(set(names)) != len(names):
            raise ValueError("Routing profile has duplicate column names")
        for name in ("rid", "token_index", "layer"):
            if name not in names:
                raise ValueError(f"Required column {name} not found in routing profile")
        expert_columns = []
        for name in names:
            if name.startswith("expert_logical_k"):
                match = re.fullmatch(r"expert_logical_k(\d+)", name)
                if match is None:
                    raise ValueError(f"Malformed expert column {name!r}")
                expert_columns.append((int(match.group(1)), name))
        expert_columns.sort()
        if len(expert_columns) != self.top_k:
            raise ValueError(f"Profile has {len(expert_columns)} expert columns; expected top_k={self.top_k}")
        if len({rank for rank, _ in expert_columns}) != len(expert_columns):
            raise ValueError("Routing profile has duplicate expert rank suffixes")

        rows = table.select(["rid", "token_index", "layer"] +
                            [name for _, name in expert_columns]).to_pylist()
        records = {}
        request_tokens = {}
        layers = set()
        for row_number, row in enumerate(rows, 1):
            rid = _nonnegative_int(row["rid"], f"row {row_number} rid")
            token = _nonnegative_int(row["token_index"], f"row {row_number} token_index")
            layer = _nonnegative_int(row["layer"], f"row {row_number} layer")
            if self.layer_id is not None and layer != self.layer_id:
                continue
            ids = tuple(_nonnegative_int(row[name], f"row {row_number} {name}")
                        for _, name in expert_columns)
            if any(expert >= self.num_experts for expert in ids):
                raise ValueError(f"row {row_number}: expert IDs must be below {self.num_experts}: {ids}")
            if len(set(ids)) != self.top_k:
                raise ValueError(f"row {row_number}: duplicate experts in Top-k routing: {ids}")
            key = (rid, token, layer)
            if key in records:
                raise ValueError(f"Duplicate routing row for rid/token/layer={key}")
            records[key] = ids
            request_tokens.setdefault(rid, set()).add(token)
            layers.add(layer)
        if not records:
            raise ValueError("Routing profile has no rows for the requested layer selection")
        expected_layers = ([self.layer_id] if self.layer_id is not None
                           else list(range(max(layers) + 1)))
        if layers != set(expected_layers):
            raise ValueError(f"Routing profile missing layers: {sorted(set(expected_layers) - layers)}")
        self.profiled_request_ids = tuple(sorted(request_tokens))
        self.profiled_token_indices = tuple(tuple(sorted(request_tokens[rid]))
                                           for rid in self.profiled_request_ids)
        self.num_profiled_requests = len(self.profiled_request_ids)
        self.num_layers = len(expected_layers)
        self.tokens_per_request = torch.tensor([len(tokens) for tokens in self.profiled_token_indices],
                                               dtype=torch.int64, device="cpu")
        self.token_prefix_sum = torch.cat((torch.zeros(1, dtype=torch.int64),
                                           self.tokens_per_request.cumsum(0)))
        self.total_tokens_per_layer = int(self.token_prefix_sum[-1])
        ordered = []
        for layer in expected_layers:
            for rid, tokens in zip(self.profiled_request_ids, self.profiled_token_indices):
                for token in tokens:
                    key = (rid, token, layer)
                    if key not in records:
                        raise ValueError(f"Routing profile missing layer row for rid/token/layer={key}")
                    ordered.append(records[key])
        self.routing_data = torch.tensor(ordered, dtype=torch.int32, device="cpu")
        self.uniform_weight = 1.0 / self.top_k

    def validate_layers(self, num_layers: int) -> None:
        """Check that a simulator's layer range is covered before it runs."""
        num_layers = _nonnegative_int(num_layers, "num_layers")
        if num_layers == 0 or num_layers > self.num_layers:
            raise ValueError(f"Simulation requests {num_layers} layers; routing profile covers {self.num_layers}")

    @staticmethod
    def _ordinals(values, label: str) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            if values.device.type != "cpu" or values.ndim != 1:
                raise ValueError(f"{label} must be a one-dimensional CPU tensor or list")
            values = values.tolist()
        try:
            values = [_nonnegative_int(value, label) for value in values]
        except TypeError as exc:
            raise ValueError(f"{label} must be a one-dimensional sequence") from exc
        return torch.tensor(values, dtype=torch.int64, device="cpu")

    def route(self, request_ids, token_indices, layer_id: int, top_k: int,
              device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return uniform weights and trace expert IDs, always on the CPU.

        ``device`` and ``dtype`` remain accepted for the runtime router's calling
        convention; CPU replay returns float32 weights and int32 expert IDs.
        """
        if top_k != self.top_k:
            raise ValueError(f"Requested top_k {top_k} differs from profile top_k {self.top_k}")
        layer_id = _nonnegative_int(layer_id, "layer_id")
        if self.layer_id is not None:
            if layer_id != self.layer_id:
                raise ValueError(f"Router is restricted to layer {self.layer_id}, got {layer_id}")
            layer_offset = 0
        else:
            if layer_id >= self.num_layers:
                raise ValueError(f"layer_id {layer_id} is outside profile layers [0, {self.num_layers})")
            layer_offset = layer_id * self.total_tokens_per_layer
        requests = self._ordinals(request_ids, "request_ids")
        tokens = self._ordinals(token_indices, "token_indices")
        if requests.shape != tokens.shape:
            raise ValueError("request_ids and token_indices must have the same length")
        mapped = requests % self.num_profiled_requests
        indices = (layer_offset + self.token_prefix_sum[mapped] +
                   tokens % self.tokens_per_request[mapped])
        ids = self.routing_data[indices]
        weights = torch.full((len(requests), self.top_k), self.uniform_weight,
                             dtype=torch.float32, device="cpu")
        return weights, ids


__all__ = ["ProfileDrivenRouter"]
