"""Exercise checkpoint-mode CLI and validation without starting Ray or CUDA."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import contextlib
import io
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from disagmoe.utils.decode_snapshot import FORMAT, checkpoint_metadata, kv_name, compare_decode_results
from experiments.scripts.rtx4090.run_local import parse_configuration


class DecodeBenchmark(unittest.TestCase):
    def test_snapshot_selects_cohort_and_rejects_graphs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = dict(model_type="qwen3_moe", num_hidden_layers=48,
                       num_key_value_heads=4, head_dim=128, vocab_size=151936)
            (root / "config.json").write_text(json.dumps(cfg))
            (root / "model.safetensors.index.json").write_text('{"weight_map":{}}')
            (root / kv_name(0)).touch()
            (root / "manifest.json").write_text(json.dumps({
                "format": FORMAT, "checkpoint_metadata_sha256": checkpoint_metadata(root),
                "repeat": 4, "prompts": [dict(input_ids=[4, 5, 6], first_token=7, expected_tokens=[8, 9])],
            }))
            argv = ["benchmark", "--checkpoint-path", temp, "--decode-snapshot", temp, "--dry-run"]
            _, args, _ = parse_configuration(argv)
            self.assertEqual((args.num_requests, args.min_input_len, args.min_output_len), (4, 3, 2))
            self.assertTrue(args.synchronized_start)
            self.assertEqual(args.generator_type, "offline")
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_configuration(argv + ["--cuda-graph-attn"])

    def test_results_are_matched_by_request_and_first_difference_is_retained(self):
        class Snapshot:
            def __len__(self):
                return 2

            def first_token(self, i):
                return 100 + i

            def expected_tokens(self, i):
                return [[11, 12], [21, 22]][i]

        snapshot = Snapshot()
        mapping = {15: 0, 28: 1}
        results = [SimpleNamespace(req_id=28, token_ids=[21, 29, 31]),
                   SimpleNamespace(req_id=15, token_ids=[11, 12, 13])]
        report = compare_decode_results(snapshot, mapping, results, 3)
        self.assertEqual(report["mismatched_requests"], 1)
        self.assertEqual(report["requests"][1]["first_different_step"], 1)
        self.assertEqual(report["requests"][0]["token_ids"], [11, 12, 13])
        with self.assertRaisesRegex(ValueError, "did not return"):
            compare_decode_results(snapshot, mapping, results, 4)
        with self.assertRaisesRegex(ValueError, "prepared cohort"):
            compare_decode_results(snapshot, mapping, results[:1], 3)


if __name__ == "__main__":
    unittest.main()
