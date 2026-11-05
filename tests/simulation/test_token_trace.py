"""Token identity and fork/join causality in observation-only async traces."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from simulation import sim_async


class TokenTraceIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.routing = self.directory / "routing.parquet"
        rows = [
            dict(rid=rid, token_index=token, layer=layer,
                 expert_logical_k0=(rid + layer) % 2,
                 expert_logical_k1=2 + (token + layer) % 2)
            for rid in range(4) for token in range(3) for layer in range(2)
        ]
        pq.write_table(pa.Table.from_pylist(rows), self.routing)
        self.profile = self.directory / "expert.csv"
        self.profile.write_text(
            "batch_size,time_ms\n" +
            "".join(f"{size},{0.06 + size * 0.01}\n" for size in range(1, 9)))

    def options(self, **overrides):
        options = dict(
            ep_group_size=2, attn_dp_group_size=2, n_gpu_per_host=2,
            total_expert_count=4, routing_top_k=2, num_layers=2,
            total_requests=4, tokens_per_request=3, arrival_rate=0,
            global_request_max_batch_size=3,
            max_batch_size=2, max_attention_batch_size=2,
            expert_profile_path=str(self.profile),
            profile_routing_path=str(self.routing),
            attn_service_t=0.2, ticks_per_millisecond=10,
            initial_context_length=0, rng_seed=72,
            net_delay_fn=lambda src, dst, copies: 0.15 + copies * 0.03,
            scheduling_policy="defrag", schedule_trace_limit=10000,
            quiet=True,
        )
        options.update(overrides)
        return options

    def run_case(self, prefix=None, **overrides):
        if prefix is not None:
            overrides["token_trace_path"] = prefix
        return sim_async.run_simulation(**self.options(**overrides))

    @staticmethod
    def records(prefix, kind="tokens"):
        with gzip.open(str(prefix) + f".{kind}.jsonl.gz", "rt") as file:
            return [json.loads(line) for line in file]

    @staticmethod
    def metadata(prefix):
        return json.loads(Path(str(prefix) + ".metadata.json").read_text())

    def assert_same_execution(self, plain, traced):
        # These cover arrivals/admission, routing, scheduling, completions,
        # accounting and measured waits; tracing may add artifact metadata.
        for key in (
            "completion_times", "makespan", "makespan_ms", "completed_tokens",
            "completed_requests", "schedule_count", "metrics_ms",
            "gpu_compute_busy_ticks", "attention_batch_histogram",
            "expert_group_batch_histogram", "active_requests_at_end",
            "waiting_requests_at_end", "ready_attention_at_end",
            "ready_expert_at_end", "incomplete_joins_at_end",
            "inflight_network_branches_at_end",
        ):
            self.assertEqual(plain[key], traced[key], key)
        self.assertEqual(len(plain["schedule_trace"]), len(traced["schedule_trace"]))
        for before, after in zip(plain["schedule_trace"], traced["schedule_trace"]):
            for key in ("gpu", "stage", "layer", "start", "end", "batch_size",
                        "expert_counts", "queue_snapshot", "scores"):
                self.assertEqual(before.get(key), after.get(key), key)

    def test_enabled_tracing_preserves_completion_and_scheduling_with_and_without_overheads(self):
        for granularity in ("layer", "expert"):
            for overhead in ("none", "structural"):
                with self.subTest(granularity=granularity, overhead=overhead):
                    options = dict(expert_scheduling=granularity,
                                   overhead_config={"preset": overhead})
                    plain = self.run_case(**options)
                    prefix = self.directory / f"{granularity}_{overhead}"
                    traced = self.run_case(prefix, **options)
                    self.assert_same_execution(plain, traced)
                    self.assertTrue(traced["token_trace"]["complete"])
                    self.assertEqual(traced["token_trace"], self.metadata(prefix))

    def test_every_token_layer_has_complete_topk_and_causal_next_attention(self):
        for granularity in ("layer", "expert"):
            with self.subTest(granularity=granularity):
                prefix = self.directory / f"causal_{granularity}"
                result = self.run_case(prefix, expert_scheduling=granularity,
                                       overhead_config={"preset": "structural"})
                rows = self.records(prefix)
                self.assertEqual(len(rows), 4 * 3 * 2)
                self.assertEqual(len({(row["token_id"], row["layer"]) for row in rows}), len(rows))
                batches = {row["batch_id"]: row for row in self.records(prefix, "batches")}
                for row in rows:
                    self.assertEqual(row["status"], "complete")
                    rid, token, layer = row["request_id"], row["token_index"], row["layer"]
                    expected = {(rid + layer) % 2, 2 + (token + layer) % 2}
                    self.assertEqual(set(row["route_expert_ids"]), expected)
                    self.assertEqual(len(row["branches"]), 2)
                    self.assertEqual({branch["expert_id"] for branch in row["branches"]}, expected)
                    source = row["source_attention"]
                    self.assertEqual(source["layer"], layer)
                    self.assertEqual(source["gpu"], row["home_gpu"])
                    self.assertIn(row["token_id"], batches[source["batch_id"]]["token_ids"])
                    self.assertIsNotNone(source["actual_completion_ms"])
                    self.assertGreaterEqual(source["actual_completion_ms"], source["compute_start_ms"])
                    for branch in row["branches"]:
                        expert = branch["expert_batch"]
                        self.assertEqual(branch["expert_gpu"], branch["expert_id"] // 2)
                        self.assertEqual(expert["gpu"], branch["expert_gpu"])
                        self.assertGreaterEqual(branch["ready_at_ms"], source["actual_completion_ms"])
                        self.assertGreaterEqual(expert["selection_ms"], branch["ready_at_ms"])
                        self.assertIsNotNone(expert["actual_completion_ms"])
                        self.assertGreaterEqual(branch["return_arrival_ms"], expert["actual_completion_ms"])
                        saved = batches[expert["batch_id"]]
                        self.assertIn(row["token_id"], saved["token_ids"])
                        self.assertIn(branch["expert_id"], saved["expert_ids"])
                    self.assertAlmostEqual(row["join_ready_ms"],
                                           max(branch["return_arrival_ms"] for branch in row["branches"]))
                    following = row["next_attention"]
                    self.assertEqual(following["layer"], layer + 1)
                    self.assertEqual(following["gpu"], row["home_gpu"])
                    self.assertAlmostEqual(following["ready_at_ms"], row["join_ready_ms"])
                    self.assertGreaterEqual(following["actual_gpu_start_ms"], row["join_ready_ms"])
                    self.assertIn(row["token_id"], batches[following["batch_id"]]["token_ids"])
                self.assertEqual(result["completed_tokens"], 12)
                # A record closes at the next stage's actual GPU start, rather
                # than treating its planned future completion as observed.
                self.assertTrue(all(row["next_attention"]["actual_completion_ms"] is None for row in rows))

    def test_window_selects_source_batches_but_follows_dependencies_past_its_end(self):
        plain = self.run_case()
        source = next(row for row in plain["schedule_trace"]
                      if row["stage"] == "attention" and row["layer"] == 1)
        start_ms = source["start"] / 10
        end_ms = start_ms + (source["end"] - source["start"]) / 100
        self.assertGreater(end_ms, start_ms)
        prefix = self.directory / "window"
        traced = self.run_case(prefix, token_trace_start_ms=start_ms,
                               token_trace_end_ms=end_ms)
        self.assert_same_execution(plain, traced)
        rows = self.records(prefix)
        self.assertTrue(rows)
        expected_sources = {(row["gpu"], row["layer"], row["start"] / 10)
                            for row in plain["schedule_trace"]
                            if row["stage"] == "attention" and
                            start_ms <= row["start"] / 10 < end_ms}
        observed_sources = set()
        for row in rows:
            self.assertEqual(row["status"], "complete")
            source = row["source_attention"]
            observed_sources.add((source["gpu"], source["layer"], source["selection_ms"]))
            self.assertGreaterEqual(source["selection_ms"], start_ms)
            self.assertLess(source["selection_ms"], end_ms)
            self.assertGreater(row["join_ready_ms"], end_ms)
            self.assertGreater(row["next_attention"]["actual_gpu_start_ms"], end_ms)
        self.assertEqual(observed_sources, expected_sources)
        # Layer-1 source members already existed before this observation window.
        self.assertTrue(any(row["layer"] == 1 for row in rows))
        self.assertTrue(traced["token_trace"]["complete"])

    def test_stream_is_independent_of_in_memory_schedule_trace_limit(self):
        prefix = self.directory / "stream_only"
        plain = self.run_case(schedule_trace_limit=0)
        traced = self.run_case(prefix, schedule_trace_limit=0)
        self.assert_same_execution(plain, traced)
        self.assertEqual(traced["schedule_trace"], [])
        self.assertEqual(len(self.records(prefix)), 4 * 3 * 2)
        self.assertGreater(len(self.records(prefix, "batches")), 0)
        self.assertTrue(traced["token_trace"]["complete"])

    def test_early_stop_marks_artifact_incomplete_and_pending_rows_censored(self):
        prefix = self.directory / "early"
        options = dict(global_request_max_batch_size=3,
                       tokens_after_reaching_max_bs=1)
        plain = self.run_case(**options)
        traced = self.run_case(prefix, **options)
        self.assert_same_execution(plain, traced)
        self.assertTrue(traced["stopped_early"])
        self.assertFalse(traced["token_trace"]["complete"])
        self.assertFalse(self.metadata(prefix)["complete"])
        rows = self.records(prefix)
        self.assertTrue(any(row["status"] == "censored" for row in rows))
        for row in rows:
            if row["status"] == "complete":
                self.assertIsNotNone(row["join_ready_ms"])
                self.assertIsNotNone(row["next_attention"]["actual_gpu_start_ms"])

    def test_execution_exception_leaves_readable_incomplete_artifact(self):
        prefix = self.directory / "failed"
        original = sim_async.AttentionCostModel.evaluate
        calls = 0

        def fail_later(model, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected token trace execution failure")
            return original(model, *args, **kwargs)

        with patch.object(sim_async.AttentionCostModel, "evaluate", new=fail_later):
            with self.assertRaisesRegex(RuntimeError, "injected token trace"):
                self.run_case(prefix)
        self.assertFalse(self.metadata(prefix)["complete"])
        # Successful gzip reads also check that exception cleanup closed both
        # compressed streams instead of leaving incomplete gzip trailers.
        self.records(prefix)
        self.records(prefix, "batches")

    def test_existing_artifacts_are_not_overwritten(self):
        prefix = self.directory / "existing"
        self.run_case(prefix)
        paths = [Path(str(prefix) + suffix) for suffix in
                 (".metadata.json", ".tokens.jsonl.gz", ".batches.jsonl.gz")]
        original = {path: path.read_bytes() for path in paths}
        with self.assertRaises(FileExistsError):
            self.run_case(prefix)
        self.assertEqual(original, {path: path.read_bytes() for path in paths})

    def test_collective_modes_reject_unsupported_token_tracing(self):
        for mode in ("sync", "tbo"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_case(self.directory / mode, execution_mode=mode)


if __name__ == "__main__":
    unittest.main()
