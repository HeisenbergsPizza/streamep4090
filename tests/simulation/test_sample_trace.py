"""Tiny local fixtures check exact trace export; no downloads or source writes."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq

from simulation.sample_trace import export_sample, sha256_file


class SampleTraceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "trace-project"
        self.cache = self.project / "data" / "processed_16fields"
        self.cache.mkdir(parents=True)
        self.array_path = self.cache / "decode_routes.npy"
        self.metadata_path = self.cache / "decode_routes_metadata.json"
        self.source_path = self.project / "download_manifest.json"
        self.array = np.empty((8, 3, 4, 2), dtype=np.uint8)
        for request in range(8):
            for token in range(3):
                for layer in range(4):
                    first = (request + token + layer) % 8
                    self.array[request, token, layer] = (first, (first + 3) % 8)
        np.save(self.array_path, self.array, allow_pickle=False)
        self.source = {"dataset": "fixture/routing", "revision": "fixture-revision",
                       "model_prefix": "Qwen/Qwen3-235B-A22B-FP8/mmlu"}
        self.source_path.write_text(json.dumps(self.source))
        self.metadata = {
            "dataset": self.source["dataset"], "manifest_revision": self.source["revision"],
            "source_manifest": "download_manifest.json", "source_manifest_sha256": sha256_file(self.source_path),
            "array_path": "data/processed_16fields/decode_routes.npy", "array_sha256": sha256_file(self.array_path),
            "array_bytes": self.array_path.stat().st_size, "shape": [8, 3, 4, 2], "dtype": "uint8",
            "axes": ["request", "decode_step", "moe_layer", "top_k_choice"],
            "request_count": 8, "decode_steps": 3, "layer_count": 4, "top_k": 2, "expert_count": 8,
            "layer_ids": [10, 11, 12, 13], "decode_entry_indices": [1, 2, 3],
            "subject_counts": {"algebra": 4, "biology": 4},
            "requests": [{"request_index": index, "inspection_request_index": index + 20,
                          "request_id": f"subject/source-{100 + index}.json",
                          "subject": "biology" if index < 4 else "algebra",
                          "source_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                          "source_relative_path": f"data/raw/{index}.json"} for index in range(8)],
        }
        self.save_metadata()

    def save_metadata(self):
        self.metadata_path.write_text(json.dumps(self.metadata))

    def save_array(self, array):
        np.save(self.array_path, array, allow_pickle=False)
        self.metadata["array_sha256"] = sha256_file(self.array_path)
        self.metadata["array_bytes"] = self.array_path.stat().st_size
        self.save_metadata()

    def export(self, name="output", **kwargs):
        options = {"requests": 4, "output_tokens": 2, "seed": 42, **kwargs}
        return export_sample(self.cache, self.root / name, **options)

    def test_balanced_sample_preserves_exact_topk_prefix_and_source_mapping(self):
        before = {path: path.read_bytes() for path in (self.array_path, self.metadata_path, self.source_path)}
        manifest = self.export(layers=2)
        self.assertEqual(manifest["source_model"], "Qwen/Qwen3-235B-A22B-FP8")
        self.assertEqual(manifest["dataset"], "fixture/routing")
        self.assertEqual(manifest["revision"], "fixture-revision")
        self.assertEqual(manifest["layer_selection"], "prefix")
        self.assertEqual(manifest["source_layer_ids"], [10, 11])
        self.assertEqual(manifest["source_decode_entry_indices"], [1, 2])
        self.assertEqual(Counter(row["subject"] for row in manifest["requests"]), {"algebra": 2, "biology": 2})
        indices = [record["cache_request_index"] for record in manifest["requests"]]
        self.assertEqual(len(set(indices)), 4)
        table = pq.read_table(manifest["routing_profile"])
        self.assertEqual(table.num_rows, 4 * 2 * 2)
        for row in table.to_pylist():
            mapping = manifest["requests"][row["rid"]]
            original = mapping["cache_request_index"]
            self.assertEqual([row["expert_logical_k0"], row["expert_logical_k1"]],
                             self.array[original, row["token_index"], row["layer"]].tolist())
            self.assertEqual(mapping["source_sha256"], self.metadata["requests"][original]["source_sha256"])
            self.assertEqual(mapping["original_request_id"], self.metadata["requests"][original]["request_id"])
        self.assertEqual(manifest["routing_sha256"], sha256_file(manifest["routing_profile"]))
        self.assertTrue(manifest["source_cache"]["sha256_verified"])
        self.assertTrue(manifest["source_manifest"]["sha256_verified"])
        self.assertFalse(manifest["synthetic"])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_seeded_replay_and_metadata_request_order_are_deterministic(self):
        first = self.export("first")
        self.metadata["requests"].reverse()
        self.save_metadata()
        second = self.export("second")
        self.assertEqual(first["requests"], second["requests"])
        self.assertEqual(first["routing_sha256"], second["routing_sha256"])
        self.assertEqual(first["source_layer_ids"], [10, 11, 12, 13])
        self.assertEqual(first["layer_selection"], "all")
        third = self.export("third", seed=8)
        self.assertNotEqual(first["requests"], third["requests"])

    def test_cache_hash_mismatch_rejected_before_output(self):
        self.metadata["array_sha256"] = "0" * 64
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "array SHA256 mismatch"):
            self.export()
        self.assertFalse((self.root / "output").exists())

    def test_source_manifest_hash_and_revision_are_checked(self):
        self.metadata["source_manifest_sha256"] = "0" * 64
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
            self.export()
        self.metadata["source_manifest_sha256"] = sha256_file(self.source_path)
        self.metadata["manifest_revision"] = "wrong-revision"
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "dataset/revision"):
            self.export()
        self.assertFalse((self.root / "output").exists())

    def test_optional_source_manifest_hash_is_honestly_marked_unverified(self):
        del self.metadata["source_manifest_sha256"]
        self.save_metadata()
        manifest = self.export()
        self.assertFalse(manifest["source_manifest"]["sha256_verified"])
        self.assertEqual(manifest["source_manifest"]["sha256"], sha256_file(self.source_path))

    def test_shape_dtype_and_topk_integrity_fail_before_writing(self):
        mutations = [
            (lambda: self.metadata.update(shape=[8, 3, 5, 2]), "shape"),
            (lambda: self.metadata.update(dtype="int64"), "dtype"),
            (lambda: self.metadata.update(axes=["wrong"] * 4), "axes"),
            (lambda: self.metadata.update(layer_ids=[0, 0, 1, 2]), "duplicate layer"),
            (lambda: self.metadata.update(decode_entry_indices=[1, 3, 4]), "contiguous"),
            (lambda: self.metadata.update(expert_id_range=[0, 3]), "outside"),
        ]
        original = json.loads(json.dumps(self.metadata))
        for mutate, expected in mutations:
            with self.subTest(expected=expected):
                self.metadata = json.loads(json.dumps(original))
                mutate()
                self.save_metadata()
                with self.assertRaisesRegex(ValueError, expected):
                    self.export()
                self.assertFalse((self.root / "output").exists())
        self.metadata = original
        bad = self.array.copy()
        bad[7, 2, 3] = [1, 1]  # outside the selected token/layer prefix
        self.save_array(bad)
        with self.assertRaisesRegex(ValueError, "duplicate experts"):
            self.export(layers=1)
        bad[7, 2, 3] = [1, 8]
        self.save_array(bad)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.export(layers=1)
        self.save_array(self.array.astype(np.float32))
        self.metadata["dtype"] = "float32"
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.export()

    def test_request_integrity_and_equal_quota_limits(self):
        for kwargs, expected in (({"requests": 3}, "divisible"), ({"requests": 10}, "insufficient"),
                                 ({"output_tokens": 4}, "exceed cache"), ({"output_tokens": 101}, "100"),
                                 ({"layers": 5}, "exceed cache"), ({"layers": 0}, "integer")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, expected):
                self.export(**kwargs)
        self.metadata["requests"][1]["request_index"] = 0
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "uniquely cover"):
            self.export()

    def test_existing_outputs_and_source_project_are_never_modified(self):
        output = self.root / "output"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "absent or empty"):
            self.export()
        self.assertEqual(sentinel.read_text(), "preserve")
        unsafe_output = self.project / "new-output"
        with self.assertRaisesRegex(ValueError, "outside the source trace project"):
            export_sample(self.cache, unsafe_output, requests=4, output_tokens=2)
        self.assertFalse(unsafe_output.exists())


if __name__ == "__main__":
    unittest.main()
