"""Small synthetic fixtures validate replay and cost contracts, not performance."""

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from simulation.costs import AttentionCostModel, ExpertCostModel
from simulation.router import ProfileDrivenRouter
from simulation.util import build_profile_router, expert_compute_time_lookup_table_from_profile


def parquet_bytes(rows):
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return sink.getvalue().to_pybytes()


def routing_rows():
    # Sparse original request/token indices, deliberately shuffled row order.
    return [
        {"rid": rid, "token_index": token, "layer": layer,
         "expert_logical_k0": first, "expert_logical_k1": second}
        for rid, token, first, second in [(90, 80, 6, 7), (10, 7, 0, 2), (10, 40, 3, 4)]
        for layer in (1, 0)
    ]


def write_csv(path, fields, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


ROUTING_INVALID_CASES = [
    (lambda rows: rows + [rows[0]], "Duplicate routing row"),
    (lambda rows: rows[:-1], "missing layer row"),
    (lambda rows: [row for row in rows if row["layer"] == 1], "missing layers"),
    (lambda rows: [dict(row, expert_logical_k0=128) for row in rows], "below 128"),
    (lambda rows: [dict(row, expert_logical_k0=-1) for row in rows], "nonnegative integer"),
    (lambda rows: [dict(row, expert_logical_k0=None) for row in rows], "nonnegative integer"),
    (lambda rows: [dict(row, expert_logical_k0=row["expert_logical_k1"]) for row in rows], "duplicate experts"),
]


class ProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def test_sparse_profile_replays_ordinals_on_cpu(self):
        path = self.tmp_path / "synthetic.parquet"
        path.write_bytes(parquet_bytes(routing_rows()))
        router = build_profile_router(str(path), num_experts=128, top_k=2)
        assert router.profiled_request_ids == (10, 90)
        assert router.profiled_token_indices == ((7, 40), (80,))
        weights, ids = router.route([0, 0, 1, 2, 3], torch.tensor([0, 1, 99, 3, 1]),
                                    layer_id=1, top_k=2, device="cuda", dtype=torch.float16)
        assert ids.tolist() == [[0, 2], [3, 4], [6, 7], [3, 4], [6, 7]]
        assert ids.device.type == weights.device.type == "cpu"
        assert ids.dtype == torch.int32
        assert weights.dtype == torch.float32
        assert torch.allclose(weights, torch.full((5, 2), 0.5))
        router.validate_layers(2)
        with self.assertRaisesRegex(ValueError, "covers 2"):
            router.validate_layers(3)


    def test_expert_columns_sort_by_numeric_suffix(self):
        row = {"rid": 100, "token_index": 55, "layer": 0}
        for k in (10, 2, 1, 0, 9, 8, 7, 6, 5, 4, 3):
            row[f"expert_logical_k{k}"] = k
        router = ProfileDrivenRouter(parquet_bytes([row]), 32, 11)
        _, ids = router.route([0], [0], 0, 11)
        assert ids.tolist() == [list(range(11))]



    def test_malformed_routing_traces_fail_early(self):
        for mutation, match in ROUTING_INVALID_CASES:
            with self.subTest(error=match), self.assertRaisesRegex(ValueError, match):
                ProfileDrivenRouter(parquet_bytes(mutation(routing_rows())), 128, 2)


    def test_router_validates_route_inputs_and_empty_batch(self):
        router = ProfileDrivenRouter(parquet_bytes(routing_rows()), 128, 2)
        weights, ids = router.route([], [], 0, 2)
        assert weights.shape == ids.shape == (0, 2)
        with self.assertRaisesRegex(ValueError, "same length"):
            router.route([0], [], 0, 2)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            router.route([0], [-1], 0, 2)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            router.route([0.5], [0], 0, 2)
        with self.assertRaisesRegex(ValueError, "outside profile"):
            router.route([0], [0], 2, 2)
        with self.assertRaisesRegex(ValueError, "differs"):
            router.route([0], [0], 0, 1)


    def test_explicit_layer_selection(self):
        router = ProfileDrivenRouter(parquet_bytes(routing_rows()), 128, 2, layer_id=1)
        assert router.num_layers == 1
        assert router.route([0], [0], 1, 2)[1].tolist() == [[0, 2]]
        with self.assertRaisesRegex(ValueError, "restricted"):
            router.route([0], [0], 0, 2)


    def test_single_expert_lookup_converts_once_and_serially_sums(self):
        path = write_csv(self.tmp_path / "expert.csv", ["batch_size", "avg_time_ms"],
                         [{"batch_size": 1, "avg_time_ms": 0.2},
                          {"batch_size": 2, "avg_time_ms": 0.3}])
        lookup = expert_compute_time_lookup_table_from_profile(path, 2, ticks_per_millisecond=10)
        cost = ExpertCostModel(lookup)
        assert math.isclose(cost([1, 0, 2, 1]), 7)
        assert math.isclose(ExpertCostModel(lookup, scale=0.5)([1, 2]), 2.5)
        assert cost([]) == cost([0, 0]) == 0
        with self.assertRaisesRegex(ValueError, "missing batch sizes"):
            cost([3])


    def test_invalid_single_expert_times_are_rejected(self):
        for time in ["nan", "inf", "-1", "0", "bad"]:
            path = write_csv(self.tmp_path / "expert.csv", ["batch_size", "avg_time_ms"],
                             [{"batch_size": 1, "avg_time_ms": time}])
            with self.subTest(time=time), self.assertRaises(ValueError):
                expert_compute_time_lookup_table_from_profile(path, 1)


    def test_lookup_requires_full_coverage_and_unique_batches(self):
        path = write_csv(self.tmp_path / "expert.csv", ["batch_size", "avg_time_ms"],
                         [{"batch_size": 1, "avg_time_ms": 0.2}])
        with self.assertRaisesRegex(ValueError, "Missing: 2"):
            expert_compute_time_lookup_table_from_profile(path, 2)
        with open(path, "a") as file:
            file.write("1,0.3\n")
        with self.assertRaisesRegex(ValueError, "Duplicate batch"):
            expert_compute_time_lookup_table_from_profile(path, 1)


    def test_grouped_profile_requires_exact_vector(self):
        path = write_csv(self.tmp_path / "grouped.csv", ["expert_counts", "time_ms"],
                         [{"expert_counts": json.dumps([1, 0, 2]), "time_ms": 0.4},
                          {"expert_counts": json.dumps([2, 0, 1]), "time_ms": 0.6}])
        cost = ExpertCostModel({1: 5, 2: 8}, grouped_profile_path=path, scale=2)
        assert math.isclose(cost([1, 0, 2]), 8)
        assert math.isclose(cost([2, 0, 1]), 12)
        with self.assertRaisesRegex(ValueError, "no exact measurement"):
            cost([1, 2])
        with self.assertRaisesRegex(ValueError, "no exact measurement"):
            cost([0, 1, 2])


    def test_attention_profile_interpolates_actual_batch_and_mean_context(self):
        path = write_csv(self.tmp_path / "attention.csv", ["batch_size", "context_length", "time_ms"],
                         [{"batch_size": 1, "context_length": 100, "time_ms": 1},
                          {"batch_size": 1, "context_length": 300, "time_ms": 3},
                          {"batch_size": 3, "context_length": 100, "time_ms": 5},
                          {"batch_size": 3, "context_length": 300, "time_ms": 9}])
        cost = AttentionCostModel(100, profile_path=path)
        # B=2 is measured between B=1 and B=3; mean history is 200.
        # Four equal interpolation weights give (1 + 3 + 5 + 9) / 4 ms.
        assert math.isclose(cost([100, 300]), 45)
        assert math.isclose(cost([200, 200]), 45)
        assert math.isclose(cost([100]), 10)
        assert cost([]) == 0
        with self.assertRaisesRegex(ValueError, "outside measured range"):
            cost([301, 301])
        with self.assertRaisesRegex(ValueError, "outside measured range"):
            cost([100] * 4)
        assert AttentionCostModel(0.5)([1, 100000]) == 1


    def test_cost_models_reject_invalid_counts(self):
        for loads in [[-1], [1.1], [True]]:
            with self.subTest(loads=loads):
                with self.assertRaisesRegex(ValueError, "nonnegative integers"):
                    ExpertCostModel({1: 1})(loads)
                with self.assertRaisesRegex(ValueError, "nonnegative integers"):
                    AttentionCostModel(1)(loads)


    def test_profiles_reject_duplicate_and_nonfinite_measurements(self):
        grouped = write_csv(self.tmp_path / "g.csv", ["expert_counts", "time_ms"],
                            [{"expert_counts": "[1]", "time_ms": 1}] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate expert_counts"):
            ExpertCostModel({}, grouped_profile_path=grouped)
        attention = write_csv(self.tmp_path / "a.csv", ["batch_size", "context_length", "time_ms"],
                              [{"batch_size": 1, "context_length": 0, "time_ms": "nan"}])
        with self.assertRaisesRegex(ValueError, "Invalid attention"):
            AttentionCostModel(1, profile_path=attention)


if __name__ == "__main__":
    unittest.main()
