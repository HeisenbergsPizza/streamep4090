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


def snapshot_cli(temp, repeat):
    root = Path(temp)
    cfg = dict(model_type="qwen3_moe", num_hidden_layers=48,
               num_key_value_heads=4, head_dim=128, vocab_size=151936)
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    (root / kv_name(0)).touch()
    (root / "manifest.json").write_text(json.dumps({
        "format": FORMAT, "checkpoint_metadata_sha256": checkpoint_metadata(root),
        "repeat": repeat, "prompts": [dict(input_ids=[4, 5, 6], first_token=7, expected_tokens=[8, 9])],
    }))
    return ["benchmark", "--checkpoint-path", temp, "--decode-snapshot", temp, "--dry-run"]


class DecodeBenchmark(unittest.TestCase):
    def test_snapshot_selects_cohort_and_rejects_graphs(self):
        with TemporaryDirectory() as temp:
            argv = snapshot_cli(temp, 4)
            _, args, _ = parse_configuration(argv)
            self.assertEqual((args.num_requests, args.min_input_len, args.min_output_len), (4, 3, 2))
            self.assertTrue(args.synchronized_start)
            self.assertEqual(args.generator_type, "offline")
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_configuration(argv + ["--cuda-graph-attn"])

    def test_paced_snapshot_has_independent_arrivals_and_batch_cap(self):
        with TemporaryDirectory() as temp:
            # Thirteen arrivals with only two rows per rank can queue across batches.
            argv = snapshot_cli(temp, 13) + ["--paced-decode", "--rate", "2.5",
                                             "--max-batch-size-attn", "2"]
            _, args, _ = parse_configuration(argv + ["--arrival-window", "1,5.2"])
            self.assertTrue(args.paced_decode)
            self.assertFalse(args.synchronized_start)
            self.assertEqual((args.num_requests, args.generator_type), (13, "uniform"))
            self.assertEqual((args.arrival_window, args.rate), ((1, 5.2), 2.5))
            for extra in (["--synchronized-start"], ["--generator-type", "offline"], ["--dp-policy", "RR"],
                          ["--analyze-throughput"], ["--arrival-window", "1,6"], ["--arrival-window", "3,3"],
                          ["--arrival-window", "1,nan"], ["--rate", "inf"], ["--rate", "0"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    parse_configuration(argv + extra)

    def test_paced_workload_requires_snapshot_and_paced_window_requires_opt_in(self):
        for extra in (["--paced-decode"], ["--arrival-window", "0,1"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_configuration(["benchmark", "--dry-run"] + extra)

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
