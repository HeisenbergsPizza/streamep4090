"""Independent arithmetic and provenance checks for the borrowed decode profile."""

import copy
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from simulation.costs import AttentionCostModel
from simulation.import_llmservingsim_attention import (
    compose_decode_rows,
    convert_bundle,
)


BUNDLE_DIR = (Path(__file__).resolve().parents[2] / "simulation" /
              "attention_costs_profiles" / "qwen3_30b_a3b_rtxpro6000_tp1")
SOURCE_DIR = BUNDLE_DIR / "source"
OUTPUT_FIELDS = ("component", "batch_size", "context_length", "time_ms")


def fixture():
    # Core T(B,L) = .01 * B * L microseconds. Dense measurements include
    # B=2, where qkv=5 rather than the value 10/3 interpolated from B=1/4.
    attention = [dict(layer="attention", prefill_chunk="0", prefill_key="0.0",
                      n_decode=str(batch), kv_decode=str(context),
                      decode_q_len="1", time_us=str(.01 * batch * context))
                 for batch in (1, 4) for context in (100, 300)]
    curves = {
        "qkv_proj": (2, 5, 6),
        "qk_norm": (1, 2, 4),
        "rotary_emb": (.5, .5, .5),
        "o_proj": (3, 4, 8),
        "layernorm": (1, 1.5, 2),
    }
    dense = [dict(layer=name, tokens=str(batch), time_us=str(time))
             for name, times in curves.items()
             for batch, time in zip((1, 2, 4), times)]
    return attention, dense


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class ComposeDecodeTests(unittest.TestCase):
    def test_component_multiplicity_units_and_dense_interior_points(self):
        attention, dense = fixture()
        original = copy.deepcopy((attention, dense))
        rows = compose_decode_rows(attention, dense)
        self.assertEqual((attention, dense), original)
        self.assertTrue(all(set(row) == set(OUTPUT_FIELDS) for row in rows))
        self.assertEqual({row["component"] for row in rows},
                         {"attention_core", "qkv_proj", "qk_norm", "rotary_emb",
                          "o_proj", "layernorm"})
        self.assertEqual({int(row["batch_size"]) for row in rows}, {1, 2, 4})
        self.assertEqual({int(row["context_length"]) for row in rows}, {100, 300})
        self.assertEqual(len(rows), 3 * 2 * 6)
        point = {row["component"]: float(row["time_ms"]) for row in rows
                 if int(row["batch_size"]) == 2 and int(row["context_length"]) == 100}
        self.assertEqual(point, {"attention_core": .002, "qkv_proj": .005,
                                 "qk_norm": .002, "rotary_emb": .0005,
                                 "o_proj": .004, "layernorm": .003})
        # A subsequent model load converts ms -> ticks exactly once. B=3,L=200
        # has 6us core + 5.5+3+.5+6+3.5us dense = 24.5us, or .245 ticks.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "composed.csv"
            write_rows(path, rows)
            model = AttentionCostModel(999, path, ticks_per_millisecond=10)
            self.assertAlmostEqual(model([100, 200, 300]), .245)
            self.assertAlmostEqual(model([100, 100]), .165)

    def test_only_q1_pure_decode_core_and_required_dense_components_contribute(self):
        attention, dense = fixture()
        expected = compose_decode_rows(attention, dense)
        extra = dict(attention[0], time_us="1000000")
        attention.extend((dict(extra, prefill_chunk="16", prefill_key="8"),
                          dict(extra, decode_q_len="2"),
                          dict(extra, layer="indexer"),
                          dict(extra, prefill_key="1"),
                          dict(extra, n_decode="0")))
        dense.extend(dict(layer=name, tokens="1", time_us="1000000")
                     for name in ("embedding", "final_layernorm", "gate_up_proj"))
        self.assertEqual(compose_decode_rows(attention, dense), expected)

    def test_order_does_not_change_output_or_drop_measured_batch_knots(self):
        attention, dense = fixture()
        self.assertEqual(compose_decode_rows(attention, dense),
                         compose_decode_rows(list(reversed(attention)), list(reversed(dense))))

    def test_missing_core_corner_duplicate_or_no_usable_rows_fail(self):
        attention, dense = fixture()
        cases = [attention[:-1], attention + [dict(attention[0])], [],
                 [dict(row, decode_q_len="2") for row in attention]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                compose_decode_rows(rows, dense)

    def test_dense_components_require_matching_axes_and_unique_points(self):
        attention, dense = fixture()
        cases = [[], dense + [dict(dense[0])],
                 [row for row in dense if row["layer"] != "o_proj"],
                 [row for row in dense
                  if not (row["layer"] == "qkv_proj" and row["tokens"] == "2")]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                compose_decode_rows(attention, rows)

    def test_outside_common_batch_range_is_not_extrapolated(self):
        attention, dense = fixture()
        # Core measurements at B=1 are outside the dense coverage [2,4].
        rows = compose_decode_rows(attention, [r for r in dense if r["tokens"] != "1"])
        self.assertEqual({int(row["batch_size"]) for row in rows}, {2, 4})
        # The two source ranges must still overlap.
        dense = [dict(row, tokens=str(int(row["tokens"]) + 10)) for row in dense]
        with self.assertRaises(ValueError):
            compose_decode_rows(attention, dense)

    def test_invalid_axes_missing_fields_and_nonfinite_costs_fail(self):
        for side, field, value in (
            ("attention", "n_decode", "1.5"),
            ("attention", "kv_decode", "-1"),
            ("attention", "kv_decode", "1.5"),
            ("dense", "tokens", "0"),
            ("dense", "tokens", "1.5"),
        ):
            attention, dense = fixture()
            (attention if side == "attention" else dense)[0][field] = value
            with self.subTest(side=side, field=field, value=value), self.assertRaises(ValueError):
                compose_decode_rows(attention, dense)
        for side in ("attention", "dense"):
            for value in ("-1", "nan", "inf", "bad"):
                attention, dense = fixture()
                (attention if side == "attention" else dense)[0]["time_us"] = value
                with self.subTest(side=side, value=value), self.assertRaises(ValueError):
                    compose_decode_rows(attention, dense)
            attention, dense = fixture()
            del (attention if side == "attention" else dense)[0]["time_us"]
            with self.subTest(side=side, missing="time_us"), self.assertRaises(ValueError):
                compose_decode_rows(attention, dense)


class BundledProfileTests(unittest.TestCase):
    def test_output_paths_cannot_replace_sources_or_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            shutil.copytree(SOURCE_DIR, source)
            originals = {path: path.read_bytes() for path in source.iterdir()}
            for path in originals:
                with self.subTest(source=path.name), self.assertRaisesRegex(ValueError, "pinned source"):
                    convert_bundle(source, path, overwrite=True)

            output = root / "attention.csv"
            manifest = output.with_suffix(".manifest.json")
            output.symlink_to(source / "attention.csv")
            with self.assertRaisesRegex(ValueError, "pinned source"):
                convert_bundle(source, output, overwrite=True)
            output.unlink()
            manifest.symlink_to(source / "dense.csv")
            with self.assertRaisesRegex(ValueError, "pinned source"):
                convert_bundle(source, output, overwrite=True)
            self.assertFalse(output.exists())
            manifest.unlink()

            # Distinct names may resolve to one destination through a symlink.
            output.write_text("existing output\n")
            manifest.symlink_to(output)
            with self.assertRaisesRegex(ValueError, "same path"):
                convert_bundle(source, output, overwrite=True)
            self.assertEqual(output.read_text(), "existing output\n")
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_changed_source_is_rejected_before_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            shutil.copytree(SOURCE_DIR, source)
            output = root / "attention.csv"
            for filename in ("attention.csv", "dense.csv", "meta.yaml"):
                path = source / filename
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "hash mismatch"):
                    convert_bundle(source, output)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".manifest.json").exists())
                path.write_bytes(original)

    def test_verified_bundle_converts_reproducibly_and_preserves_batch_economy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attention.csv"
            manifest = convert_bundle(SOURCE_DIR, output)
            self.assertIsInstance(manifest, dict)
            manifest_path = output.with_suffix(".manifest.json")
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)
            original = output.read_bytes()
            self.assertEqual(original, (BUNDLE_DIR / "attention.csv").read_bytes())
            second = Path(directory) / "second.csv"
            convert_bundle(SOURCE_DIR, second)
            self.assertEqual(original, second.read_bytes())
            with self.assertRaises((FileExistsError, ValueError)):
                convert_bundle(SOURCE_DIR, output)
            self.assertEqual(original, output.read_bytes())
            convert_bundle(SOURCE_DIR, output, overwrite=True)
            self.assertEqual(original, output.read_bytes())

            model = AttentionCostModel(999, output, ticks_per_millisecond=1)
            for batch in range(1, 257):
                for context in (1024, 1055):
                    with self.subTest(batch=batch, context=context):
                        estimate = model.evaluate([context] * batch)
                        self.assertTrue(math.isfinite(estimate.total_ticks))
                        self.assertGreater(estimate.total_ticks, 0)
            large_us = model([1024] * 128) * 1000
            small_us = model([1024] * 8) * 1000
            self.assertAlmostEqual(large_us, 273.0561009978308, places=7)
            self.assertAlmostEqual(small_us, 60.35767707158351, places=7)
            # Same 128 token-layer work, split into sixteen small operations.
            self.assertGreater(16 * small_us, 3 * large_us)
            self.assertLess(16 * small_us, 4 * large_us)
            with self.assertRaises(ValueError):
                model([1024] * 257)


if __name__ == "__main__":
    unittest.main()
