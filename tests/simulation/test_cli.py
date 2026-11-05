"""Artifact-level tests of the public CPU runner."""
import contextlib
import csv
import io
import json
import gzip
from pathlib import Path
import tempfile
import unittest

from simulation.run import main


class CLITests(unittest.TestCase):
    def test_decode_attention_profile_is_used_and_recorded_in_all_modes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route, expert, attention = (root / name for name in ('route.parquet', 'expert.csv', 'attention.csv'))
            pq.write_table(pa.Table.from_pylist([
                dict(rid=0, token_index=t, layer=0, expert_logical_k0=0, expert_logical_k1=1)
                for t in (0, 1)]), route)
            expert.write_text('batch_size,time_ms\n1,0.01\n2,0.02\n')
            attention.write_text('component,batch_size,context_length,time_ms\n'
                                 'qkv,1,10,0.02\nattention_core,1,10,0.03\n'
                                 'qkv,1,12,0.02\nattention_core,1,12,0.13\n')
            for mode in ('async', 'sync', 'tbo'):
                with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()):
                    output = root / mode
                    main(['--mode', mode, '--routing-profile', str(route),
                          '--expert-profile', str(expert), '--attention-profile', str(attention),
                          '--gpus', '1', '--experts', '2', '--top-k', '2', '--layers', '1',
                          '--requests', '1', '--output-tokens', '2', '--active-request-cap', '1',
                          '--attention-batch-cap', '2', '--expert-batch-cap', '2',
                          '--initial-context', '10', '--output-dir', str(output)])
                    label = 'async_defrag' if mode == 'async' else mode
                    data = json.loads((output / f'{label}.json').read_text())
                    self.assertAlmostEqual(data['simulation']['makespan'], 1.9)
                    self.assertEqual(data['simulation']['attention_cost_model']['components'],
                                     ['attention_core', 'qkv'])
                    self.assertEqual(data['provenance']['cost_assumptions']['attention'],
                                     'decode_batch_context_profile')
                    self.assertEqual(data['provenance']['attention']['path'], str(attention.resolve()))

    def test_expert_scheduling_rejects_baselines_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("sync", "tbo"):
                output, error = Path(directory) / mode, io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                    main(["--synthetic", "--mode", mode, "--expert-scheduling", "expert",
                          "--output-dir", str(output)])
                self.assertIn("--expert-scheduling expert applies only to --mode async", error.getvalue())
                self.assertFalse(output.exists())

    def test_expert_scheduling_runs_all_policies_and_records_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "expert-wise"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--synthetic", "--expert-scheduling", "expert",
                      "--policies", "defrag,flfs,mtfs,fifo", "--gpus", "2",
                      "--experts", "4", "--top-k", "2", "--layers", "2",
                      "--requests", "4", "--output-tokens", "2", "--active-request-cap", "4",
                      "--attention-batch-cap", "4", "--expert-batch-cap", "4",
                      "--schedule-trace-limit", "1000", "--output-dir", str(output)])
            manifest = json.loads((output / "inputs.json").read_text())
            self.assertEqual(manifest["configuration"]["expert_scheduling"], "expert")
            reference = manifest["provenance"]["scheduling"]
            self.assertEqual(reference["runtime_expert_scheduling"], "layer")
            self.assertTrue(reference["simulation_only_extension"])
            self.assertEqual(reference["expert_batch_cap_scope"], "selected_gpu_layer_single_expert_branches")
            for policy in ("defrag", "flfs", "mtfs", "fifo"):
                result = json.loads((output / f"async_{policy}.json").read_text())
                self.assertEqual(result["summary"]["expert_scheduling"], "expert")
                self.assertEqual(result["summary"]["completed_tokens"], 8)
                self.assertEqual(result["simulation"]["expert_scheduling"], "expert")
                experts = [item for item in result["simulation"]["schedule_trace"]
                           if item["stage"] == "expert"]
                self.assertTrue(experts)
                for item in experts:
                    self.assertIsInstance(item["expert_id"], int)
                    self.assertEqual(sum(count > 0 for count in item["expert_counts"]), 1)

    def test_supplied_profiles_cannot_inherit_smoke_batch_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route, cost = root / 'routing.parquet', root / 'cost.csv'
            route.write_text('not read before argument validation')
            cost.write_text('not read before argument validation')
            for extra in ([], ['--attention-batch-cap', '256'], ['--expert-batch-cap', '1024']):
                output, error = root / 'output', io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                    main(['--routing-profile', str(route), '--expert-profile', str(cost),
                          '--output-dir', str(output), *extra])
                self.assertIn('require explicit --attention-batch-cap and --expert-batch-cap', error.getvalue())
                self.assertFalse(output.exists())

    def test_queue_trace_writes_observation_artifacts_independent_of_decision_limit(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "traced"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--synthetic", "--gpus", "2", "--experts", "4", "--top-k", "2",
                      "--layers", "2", "--requests", "4", "--output-tokens", "3",
                      "--active-request-cap", "4", "--expert-batch-cap", "4",
                      "--queue-trace", "--queue-trace-interval-ms", "0.02",
                      "--queue-trace-start-ms", "0.01", "--queue-trace-end-ms", "0.123",
                      "--schedule-trace-limit", "1", "--output-dir", str(out)])
            prefix = out / "async_defrag_queue_trace"
            self.assertTrue(prefix.with_suffix(".metadata.json").is_file())
            with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as data:
                self.assertAlmostEqual(data["time_edges_ms"][0], 0.01)
                self.assertAlmostEqual(data["time_edges_ms"][-1], 0.123)
                self.assertEqual(data["queue_mean"].shape[1:], (2, 5))
                self.assertTrue(np.all(data["queue_mean"] >= 0))
            with gzip.open(prefix.with_suffix(".decisions.jsonl.gz"), "rt") as file:
                decisions = [json.loads(line) for line in file]
            self.assertGreater(len(decisions), 1)
            result = json.loads((out / "async_defrag.json").read_text())["simulation"]
            self.assertEqual(len(result["schedule_trace"]), 1)
            self.assertEqual(result["completed_tokens"], 12)
            self.assertIn("queue_trace", result)

    def test_invalid_queue_trace_options_fail_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for invalid in (["--mode", "tbo", "--queue-trace"],
                            ["--queue-trace-interval-ms", "0"],
                            ["--queue-trace-start-ms", "2", "--queue-trace-end-ms", "1"]):
                out = Path(directory) / "invalid"
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["--synthetic", "--output-dir", str(out), *invalid])
                self.assertFalse(out.exists())

    def test_all_modes_write_consistent_counts_and_adjacent_itl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("async", "sync", "tbo"):
                with self.subTest(mode=mode):
                    out = root / mode
                    args = ["--synthetic", "--mode", mode, "--gpus", "2",
                            "--experts", "4", "--top-k", "2", "--layers", "2",
                            "--requests", "4", "--output-tokens", "3",
                            "--active-request-cap", "4", "--attention-batch-cap", "4",
                            "--expert-batch-cap", "4", "--output-dir", str(out)]
                    if mode == "async":
                        args += ["--policies", "flfs,mtfs,fifo,defrag"]
                    with contextlib.redirect_stdout(io.StringIO()):
                        main(args)
                    manifest = json.loads((out / "inputs.json").read_text())
                    self.assertEqual(manifest["provenance"]["input_kind"], "synthetic_fixture")
                    self.assertFalse(manifest["provenance"]["hardware_calibrated"])
                    self.assertEqual(manifest["configuration"]["expert_scheduling"], "layer")
                    self.assertFalse(manifest["provenance"]["scheduling"]["simulation_only_extension"])
                    with (out / "comparison.csv").open() as f:
                        summaries = list(csv.DictReader(f))
                    self.assertEqual(len(summaries), 4 if mode == "async" else 1)
                    for summary in summaries:
                        with Path(summary["token_csv"]).open() as f:
                            tokens = list(csv.DictReader(f))
                        self.assertEqual(int(summary["completed_tokens"]), len(tokens))
                        self.assertEqual(len(tokens), 12)
                        self.assertEqual(int(summary["completed_requests"]), 4)
                        self.assertEqual(int(summary["itl_count"]), 8)
                        previous = {}
                        for row in tokens:
                            rid = int(row["request_id"])
                            now = float(row["completion_ticks"])
                            if int(row["token_index"]) == 0:
                                self.assertEqual(row["itl_ms"], "")
                            else:
                                self.assertAlmostEqual(float(row["itl_ms"]), (now - previous[rid]) / 10)
                            previous[rid] = now
                        label = f"async_{summary['policy']}" if mode == "async" else mode
                        data = json.loads((out / f"{label}.json").read_text(),
                                          parse_constant=lambda value: self.fail(f"nonstandard JSON {value}"))
                        self.assertEqual(data["provenance"]["routing"]["sha256"],
                                         manifest["provenance"]["routing"]["sha256"])

    def test_nonempty_output_is_preserved_and_synthetic_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            sentinel = out / "existing.txt"
            sentinel.write_text("preserve")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--synthetic", "--output-dir", str(out)])
            self.assertEqual(sentinel.read_text(), "preserve")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--output-dir", str(out / "missing-profiles")])
            self.assertFalse((out / "missing-profiles").exists())

    def test_early_stop_without_completed_requests_leaves_missing_latency_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("async", "sync", "tbo"):
                with self.subTest(mode=mode):
                    out = Path(directory) / mode
                    with contextlib.redirect_stdout(io.StringIO()):
                        main(["--synthetic", "--mode", mode, "--gpus", "2",
                              "--experts", "4", "--top-k", "2", "--layers", "2",
                              "--requests", "4", "--output-tokens", "4",
                              "--active-request-cap", "4", "--expert-batch-cap", "4",
                              "--tokens-after-saturation", "1", "--output-dir", str(out)])
                    with (out / "comparison.csv").open() as file:
                        summary = next(csv.DictReader(file))
                    self.assertGreater(int(summary["completed_tokens"]), 0)
                    self.assertEqual(int(summary["completed_requests"]), 0)
                    self.assertEqual(summary["avg_request_latency_ms"], "")
                    self.assertEqual(summary["itl_mean_ms"], "")
                    label = "async_defrag" if mode == "async" else mode
                    data = json.loads((out / f"{label}.json").read_text(),
                                      parse_constant=lambda value: self.fail(f"nonstandard JSON {value}"))
                    self.assertIsNone(data["summary"]["avg_request_latency_ms"])
                    self.assertEqual(data["itl_ms"]["count"], 0)
                    self.assertTrue(data["simulation"]["stopped_early"])


if __name__ == "__main__":
    unittest.main()
