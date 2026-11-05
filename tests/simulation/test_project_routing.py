"""Check the opt-in routing derivation without changing simulator behavior."""

import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from simulation.project_routing import main, project_routing
from simulation.router import ProfileDrivenRouter


class ProjectRoutingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "source.parquet"
        self.rows = []
        for rid in (4, 9, 15, 28):
            for token in (2, 5, 11):
                for layer in range(3):
                    row = {"rid": rid, "token_index": token, "layer": layer}
                    row.update({
                        f"expert_logical_k{rank}": (rid + token + layer + rank) % 16
                        for rank in range(8)
                    })
                    self.rows.append(row)
        self.write(self.rows)

    def write(self, rows):
        pq.write_table(pa.Table.from_pylist(rows), self.input)

    def project(self, name="output", **kwargs):
        return project_routing(self.input, self.root / name, experts=16, **kwargs)

    def test_member_keys_source_and_manifest_preserved(self):
        original = self.input.read_bytes()
        manifest = self.project(seed=42)
        table = pq.read_table(manifest["routing_profile"])
        self.assertEqual(table.column_names,
                         ["rid", "token_index", "layer", "expert_logical_k0"])
        self.assertEqual(table.num_rows, len(self.rows))
        for source, projected in zip(self.rows, table.to_pylist()):
            self.assertEqual(
                [source[name] for name in ("rid", "token_index", "layer")],
                [projected[name] for name in ("rid", "token_index", "layer")])
            self.assertIn(projected["expert_logical_k0"],
                          [source[f"expert_logical_k{rank}"] for rank in range(8)])
        self.assertEqual(original, self.input.read_bytes())
        self.assertEqual(manifest["source_routing"]["sha256"],
                         hashlib.sha256(original).hexdigest())
        self.assertEqual(manifest["source_top_k"], 8)
        self.assertEqual(manifest["top_k"], 1)
        self.assertEqual(manifest["layer_count"], 3)
        self.assertEqual(manifest["request_count"], 4)
        self.assertTrue(manifest["evidence_boundary"]["counterfactual_routing"])
        self.assertFalse(manifest["evidence_boundary"]["native_top1_trace"])
        self.assertFalse(manifest["evidence_boundary"]["hardware_calibrated"])
        self.assertEqual(manifest["statistics"]["before_assignments"], 8 * len(self.rows))
        self.assertEqual(manifest["statistics"]["after_assignments"], len(self.rows))
        stats = manifest["statistics"]["per_expert_assignment_counts"]
        self.assertEqual(sum(item["before"] for item in stats), 8 * len(self.rows))
        self.assertEqual(sum(item["after"] for item in stats), len(self.rows))
        self.assertEqual(json.loads((self.root / "output" / "projection_manifest.json").read_text()),
                         manifest)
        router = ProfileDrivenRouter(Path(manifest["routing_profile"]).read_bytes(), 16, 1)
        self.assertEqual(router.num_layers, 3)

    def test_seed_repeat_and_row_order_independent_selection(self):
        first = self.project("first", seed=19)
        second = self.project("second", seed=19)
        self.assertEqual(first["routing_sha256"], second["routing_sha256"])
        random.Random(6).shuffle(self.rows)
        self.write(self.rows)
        reordered = self.project("reordered", seed=19)
        # Source hash metadata changes after physically reordering input rows.
        self.assertTrue(pq.read_table(first["routing_profile"]).equals(
            pq.read_table(reordered["routing_profile"]), check_metadata=False))
        different = self.project("different", seed=20)
        self.assertNotEqual(
            pq.read_table(first["routing_profile"]).to_pylist(),
            pq.read_table(different["routing_profile"]).to_pylist())

    def test_uniform_rank_draws_and_exact_counts(self):
        rows = []
        for token in range(4096):
            row = {"rid": 0, "token_index": token, "layer": 0}
            row.update({f"expert_logical_k{rank}": rank for rank in range(8)})
            rows.append(row)
        self.write(rows)
        manifest = self.project(seed=31)
        counts = manifest["statistics"]["selected_source_rank_counts"]
        self.assertEqual(sum(counts.values()), len(rows))
        # A deliberately broad deterministic check catches rank bias and always
        # choosing one rank; it is not a statistical quality certification.
        self.assertTrue(all(350 < count < 700 for count in counts.values()))
        expert_counts = manifest["statistics"]["per_expert_assignment_counts"]
        self.assertEqual([item["after"] for item in expert_counts[:8]],
                         [counts[str(rank)] for rank in range(8)])
        self.assertTrue(all(item["after"] == 0 for item in expert_counts[8:]))

    def test_top1_source_and_extra_columns_supported(self):
        rows = [{**{name: value for name, value in row.items()
                    if not name.startswith("expert_logical_k") or name == "expert_logical_k0"},
                 "note": "source annotation"} for row in self.rows]
        self.write(rows)
        manifest = self.project()
        self.assertEqual(manifest["source_top_k"], 1)
        self.assertEqual(manifest["preservation"]["dropped_columns"], ["note"])
        self.assertEqual(manifest["statistics"]["selected_source_rank_counts"], {"0": len(rows)})
        self.assertEqual(pq.read_table(manifest["routing_profile"])["expert_logical_k0"].to_pylist(),
                         [row["expert_logical_k0"] for row in rows])

    def test_input_validation_rejects_invalid_keys_experts_and_layer_coverage(self):
        def replace(row, key, value):
            rows = [dict(item) for item in self.rows]
            rows[row][key] = value
            return rows

        cases = [
            (self.rows + [self.rows[0]], "Duplicate routing row"),
            (replace(0, "rid", -1), "nonnegative integer"),
            (replace(0, "token_index", 1.5), "nonnegative integer"),
            (replace(0, "layer", None), "nonnegative integer"),
            (replace(0, "expert_logical_k0", 16), "below 16"),
            (replace(0, "expert_logical_k0", self.rows[0]["expert_logical_k1"]),
             "duplicate experts"),
            (self.rows[1:], "missing layer row"),
            ([row for row in self.rows if row["layer"] != 1], "missing layers"),
        ]
        for rows, message in cases:
            with self.subTest(message=message):
                self.write(rows)
                with self.assertRaisesRegex(ValueError, message):
                    self.project()
                self.assertFalse((self.root / "output").exists())

    def test_schema_validation(self):
        original = pa.Table.from_pylist(self.rows)
        cases = [
            (original.drop(["rid"]), "Required column rid"),
            (original.drop(["expert_logical_k2"]), "contiguous"),
            (original.rename_columns(
                ["expert_logical_k00" if name == "expert_logical_k0" else name
                 for name in original.column_names]), "Malformed"),
            (original.drop([f"expert_logical_k{rank}" for rank in range(8)]), "contiguous"),
            (original.slice(0, 0), "no rows"),
        ]
        for table, message in cases:
            with self.subTest(message=message):
                pq.write_table(table, self.input)
                with self.assertRaisesRegex(ValueError, message):
                    self.project()
                self.assertFalse((self.root / "output").exists())
        self.input.write_bytes(b"not parquet")
        with self.assertRaisesRegex(ValueError, "Failed to parse"):
            self.project()

    def test_source_manifest_hash_and_optional_reference(self):
        source_manifest = self.root / "sample_manifest.json"
        source_manifest.write_text(json.dumps({
            "source_model": "fixture/model",
            "routing_sha256": hashlib.sha256(self.input.read_bytes()).hexdigest(),
        }))
        original = source_manifest.read_bytes()
        manifest = self.project(source_manifest=source_manifest)
        self.assertTrue(manifest["source_manifest"]["routing_sha256_verified"])
        self.assertEqual(manifest["source_manifest"]["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(source_manifest.read_bytes(), original)
        source_manifest.write_text(json.dumps({"routing_sha256": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.project("bad", source_manifest=source_manifest)
        self.assertFalse((self.root / "bad").exists())
        source_manifest.write_text("{}")
        manifest = self.project("reference", source_manifest=source_manifest)
        self.assertFalse(manifest["source_manifest"]["routing_sha256_verified"])

    def test_existing_output_invalid_mode_and_scalar_arguments_rejected(self):
        self.project()
        output = self.root / "output"
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaisesRegex(ValueError, "absent or empty"):
            self.project()
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
        for kwargs in ({"mode": "highest-score"}, {"seed": -1}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.project("bad", **kwargs)
        with self.assertRaisesRegex(ValueError, "experts"):
            project_routing(self.input, self.root / "bad", experts=0)
        self.assertFalse((self.root / "bad").exists())

    def test_cli_writes_expected_artifacts(self):
        main(["--input", str(self.input), "--output-dir", str(self.root / "cli"),
              "--mode", "uniform-from-topk", "--seed", "23", "--experts", "16"])
        manifest = json.loads((self.root / "cli" / "projection_manifest.json").read_text())
        self.assertEqual(manifest["seed"], 23)
        self.assertTrue((self.root / "cli" / "routing.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
