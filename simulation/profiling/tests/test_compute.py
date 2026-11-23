"""Portable checks of profile coordinates and the actual simulator ingestion contracts."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simulation.costs import AttentionCostModel, ExpertCostModel
from simulation.profiling import compute
from simulation.util import expert_compute_time_lookup_table_from_profile


class ComputeProfileTest(unittest.TestCase):
    def test_synthetic_vectors_respect_local_counts(self):
        vectors = compute.vector_workload(None, "1,8,16", 8, 16)
        self.assertIn((16, 0, 0, 0, 0, 0, 0, 0), vectors)
        self.assertIn((4, 4, 4, 4, 0, 0, 0, 0), vectors)
        self.assertIn((2,) * 8, vectors)
        self.assertTrue(all(sum(x) in (1, 8, 16) for x in vectors))
        self.assertTrue(all(n == 1 for n in vectors.values()))

    def test_exact_vectors_from_mixed_trace(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl.gz"
            rows = [{"stage": "attention", "expert_counts": None}, {"stage": "finalize"},
                    {"stage": "expert", "expert_counts": [0, 3]}, [3, 0], [0, 3]]
            with gzip.open(path, "wt") as f:
                f.writelines(json.dumps(x) + "\n" for x in rows)
            self.assertEqual(compute.vector_workload(path, None, 2, 3),
                             {(0, 3): 2, (3, 0): 1})

    def test_bad_vector_reports_input_line(self):
        for invalid in ([True, 0], [0, 0], [1, 2, 3], [0, -1], [4, 0], [1.2, 0]):
            with self.subTest(invalid=invalid), TemporaryDirectory() as temp:
                path = Path(temp) / "raw.jsonl"
                path.write_text("[1, 0]\n" + json.dumps(invalid) + "\n")
                with self.assertRaisesRegex(ValueError, r"raw.jsonl:2"):
                    compute.vector_workload(path, None, 2, 3)

    def test_native_graph_cap(self):
        for value in (1, 96, 128, 256, 512):
            compute.validate_graph_expert_cap(value)
        with self.assertRaises(ValueError):
            compute.validate_graph_expert_cap(300)

    def test_full_decode_kv_grid(self):
        self.assertEqual([compute.blocks_per_seq(x, 16) for x in (0, 15, 16, 63, 64)],
                         [1, 1, 2, 4, 5])
        self.assertEqual(compute.integer_grid("64,0,128", minimum=0), [0, 64, 128])
        with self.assertRaises(ValueError):
            compute.integer_grid("1,1")

    def test_tables_fit_simulator_readers(self):
        args = SimpleNamespace(warmup=2, repeats=3)
        grids = [
            ("expert", ["batch_size"], [dict(batch_size=i) for i in range(1, 4)]),
            ("grouped", ["expert_counts"], [dict(expert_counts=(0, 3), occurrences=2)]),
            ("attention", ["batch_size", "context_length"],
             [dict(batch_size=b, context_length=l) for b in (1, 4) for l in (0, 64)]),
        ]

        with TemporaryDirectory() as temp, patch.object(compute, "measure_cuda", return_value=[3, 1, 2]) as timer:
            root = Path(temp)
            for stage, columns, cases in grids:
                out = root / stage
                out.mkdir()
                compute.record_grid(out, stage, columns, cases, lambda case: (lambda: None, {}), args)
                with (out / f"{stage}.csv").open() as f:
                    self.assertTrue(all(float(row["time_ms"]) == 2 for row in csv.DictReader(f)))
                raw = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
                self.assertTrue(all(row["samples_ms"] == [3, 1, 2] for row in raw))
                self.assertEqual(timer.call_args.args[1:], (args.warmup, args.repeats))
            lookup = expert_compute_time_lookup_table_from_profile(
                root / "expert" / "expert.csv", max_batch_size=3, ticks_per_millisecond=1)
            self.assertEqual([lookup[n] for n in range(1, 4)], [2, 2, 2])
            self.assertEqual(ExpertCostModel(lookup, grouped_profile_path=root / "grouped/grouped.csv",
                                             ticks_per_millisecond=1)([0, 3]), 2)
            self.assertEqual(AttentionCostModel(1, root / "attention/attention.csv", 1)([32, 32]), 2)

    def test_cli_without_cuda_runtime(self):
        parser = argparse.ArgumentParser()
        compute.add_commands(parser.add_subparsers(required=True))
        for stage in ("expert", "grouped", "attention"):
            args = parser.parse_args([stage, "--bundle", "sample"])
            self.assertEqual(args.stage, stage)
            self.assertTrue(callable(args.func))
