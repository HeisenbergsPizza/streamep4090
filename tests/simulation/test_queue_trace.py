"""Exact queue integrals and observation-only simulator tracing."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from simulation.queue_trace import QueueTraceRecorder
from simulation import sim_async
from simulation.sim_async import run_simulation


class _QueueTraceTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.trace_number = 0

    def recorder(self, **overrides):
        options = dict(num_gpus=1, num_layers=2, ticks_per_millisecond=1,
                       interval_ms=1)
        options.update(overrides)
        self.trace_number += 1
        recorder = QueueTraceRecorder(self.directory / f"trace_{self.trace_number}", **options)
        self.addCleanup(recorder.close)
        return recorder

    def arrays(self, metadata):
        with np.load(metadata["paths"]["npz"], allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}


class QueueTraceTests(_QueueTraceTestCase):
    def test_exact_integral_union_and_finalize_exclusion(self):
        trace = self.recorder()
        trace.change(0, 0, 0, 2)       # A0
        trace.change(0.25, 0, 3, 3)   # E0: same logical layer as A0
        trace.change(0.5, 0, 1, 1)    # A1
        trace.change(0.75, 0, 0, -2)
        trace.change(1, 0, 1, -1)
        trace.change(1, 0, 2, 7)      # finalize, excluded from active statistics
        metadata = trace.finish(1.5)
        data = self.arrays(metadata)
        np.testing.assert_allclose(data["time_edges_ms"], [0, 1, 1.5])
        np.testing.assert_allclose(data["queue_mean"][:, 0],
                                   [[1.5, 0.5, 0, 2.25, 0], [0, 0, 7, 3, 0]])
        np.testing.assert_allclose(data["active_layers_mean"][:, 0], [1.5, 1])
        np.testing.assert_allclose(data["attention_active_layers_mean"][:, 0], [1.25, 0])
        np.testing.assert_allclose(data["expert_active_layers_mean"][:, 0], [0.75, 1])
        self.assertEqual(metadata["observed_window_ms"], {"start": 0, "end": 1.5})
        self.assertEqual(json.loads(Path(metadata["paths"]["metadata"]).read_text()), metadata)

    def test_window_clip_spanning_bins_and_idle_tail(self):
        trace = self.recorder(num_layers=1, start_ms=1, end_ms=3.4)
        trace.change(0, 0, 0, 2)
        trace.change(1.5, 0, 0, 4)
        trace.change(2.5, 0, 0, -6)
        metadata = trace.finish(5)
        data = self.arrays(metadata)
        np.testing.assert_allclose(data["time_edges_ms"], [1, 2, 3, 3.4])
        np.testing.assert_allclose(data["queue_mean"][:, 0, 0], [4, 3, 0])
        np.testing.assert_allclose(data["active_layers_mean"][:, 0], [1, 0.5, 0])
        self.assertEqual(metadata["simulation_end_ms"], 5)
        self.assertAlmostEqual(metadata["observed_duration_ms"], 2.4)

    def test_same_timestamp_changes_have_no_duration_and_partial_bin_uses_actual_time(self):
        trace = self.recorder(ticks_per_millisecond=10)
        trace.change(0, 0, 0, 100)
        trace.change(0, 0, 0, -100)
        trace.change(0, 0, 1, 3)
        trace.change(2.5, 0, 1, -3)
        metadata = trace.finish(2.5)
        data = self.arrays(metadata)
        np.testing.assert_allclose(data["time_edges_ms"], [0, 0.25])
        np.testing.assert_allclose(data["queue_mean"][0, 0], [0, 3, 0, 0, 0])

    def test_idle_observation_and_unobserved_window_are_distinct(self):
        idle = self.recorder(num_gpus=2)
        metadata = idle.finish(2.5)
        data = self.arrays(metadata)
        self.assertEqual(data["queue_mean"].shape, (3, 2, 5))
        self.assertEqual(np.count_nonzero(data["queue_mean"]), 0)
        np.testing.assert_allclose(data["time_edges_ms"], [0, 1, 2, 2.5])
        empty = self.recorder(start_ms=4)
        metadata = empty.finish(2.5)
        data = self.arrays(metadata)
        self.assertEqual(data["queue_mean"].shape, (0, 1, 5))
        self.assertEqual(data["active_layers_mean"].shape, (0, 1))
        np.testing.assert_array_equal(data["time_edges_ms"], [4])
        self.assertEqual(metadata["observed_window_ms"], {"start": None, "end": None})
        self.assertEqual(metadata["observed_duration_ms"], 0)

    def test_decisions_include_carry_in_and_zero_cost_but_exclude_end_boundary(self):
        trace = self.recorder(start_ms=1, end_ms=3, ticks_per_millisecond=10)
        for start, end in ((5, 10), (9, 11), (10, 10), (25, 100), (30, 100), (30, 30)):
            trace.capture_schedule(dict(start=start, end=end, gpu=0, stage="attention",
                                        layer=0, batch_size=1, expert_counts=None,
                                        queue_snapshot=[1, 0, 0, 0, 0], scores=[1, 0, 0, 0, 0]))
        metadata = trace.finish(27)
        with gzip.open(metadata["paths"]["decisions"], "rt") as handle:
            decisions = [json.loads(line) for line in handle]
        self.assertEqual([row["start"] for row in decisions], [9, 10, 25])
        self.assertEqual([row["start_ms"] for row in decisions], [0.9, 1, 2.5])
        self.assertEqual([row["end_ms"] for row in decisions], [1.1, 1, 10])
        self.assertEqual(metadata["decision_count"], 3)
        self.assertEqual(metadata["observed_window_ms"]["end"], 2.7)

    def test_existing_prefix_is_preserved_and_incomplete_metadata_is_explicit(self):
        trace = self.recorder()
        metadata_path = Path(trace.paths["metadata"])
        incomplete = metadata_path.read_text()
        self.assertFalse(json.loads(incomplete)["complete"])
        with self.assertRaises(FileExistsError):
            QueueTraceRecorder(self.directory / "trace_1", num_gpus=1, num_layers=2)
        self.assertEqual(metadata_path.read_text(), incomplete)
        self.assertTrue(trace.finish(1)["complete"])

    def test_invalid_clock_counts_and_options_fail(self):
        trace = self.recorder()
        with self.assertRaisesRegex(ValueError, "negative"):
            trace.change(0, 0, 0, -1)
        trace.change(1, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            trace.change(0.5, 0, 0, 1)
        for options in ({"interval_ms": 0}, {"start_ms": -1}, {"end_ms": 0},
                        {"end_ms": float("inf")}, {"ticks_per_millisecond": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.recorder(**options)


class QueueTraceIntegrationTests(_QueueTraceTestCase):
    def options(self, **overrides):
        trace = self.directory / "routes.parquet"
        rows = [dict(rid=rid, token_index=token, layer=layer,
                     expert_logical_k0=(rid + layer) % 2,
                     expert_logical_k1=2 + (token + layer) % 2)
                for rid in range(4) for token in range(3) for layer in range(2)]
        pq.write_table(pa.Table.from_pylist(rows), trace)
        profile = self.directory / "experts.csv"
        profile.write_text("batch_size,time_ms\n1,0.11\n2,0.13\n3,0.17\n4,0.2\n")
        options = dict(ep_group_size=2, attn_dp_group_size=2, total_expert_count=4,
                       routing_top_k=2, num_layers=2, total_requests=4,
                       tokens_per_request=3, arrival_rate=0,
                       max_batch_size=4, max_attention_batch_size=4,
                       global_request_max_batch_size=3, attn_service_t=0.5,
                       expert_profile_path=str(profile), profile_routing_path=str(trace),
                       schedule_trace_limit=10000, metric_sample_capacity=2,
                       rng_seed=72, quiet=True)
        options.update(overrides)
        return options

    def test_tracing_preserves_every_existing_result_and_decision(self):
        for policy in ("defrag", "fifo", "mtfs", "flfs"):
            with self.subTest(policy=policy):
                options = self.options(scheduling_policy=policy,
                                       net_delay_fn=lambda src, dst, copies: 0.2 + copies * 0.01)
                plain = run_simulation(**options)
                traced = run_simulation(**options, queue_trace_path=self.directory / policy,
                                        queue_trace_interval_ms=0.3)
                metadata = traced.pop("queue_trace")
                self.assertEqual(json.dumps(plain, sort_keys=True), json.dumps(traced, sort_keys=True))
                data = self.arrays(metadata)
                self.assertAlmostEqual(data["time_edges_ms"][-1], traced["makespan_ms"])
                self.assertEqual(metadata["policy"], policy)
                self.assertEqual(metadata["top_k"], 2)
                self.assertFalse(metadata["hardware_calibrated"])
                with gzip.open(metadata["paths"]["decisions"], "rt") as handle:
                    decisions = [json.loads(line) for line in handle]
                self.assertEqual(len(decisions), traced["schedule_count"])
                for saved, original in zip(decisions, traced["schedule_trace"]):
                    for key, value in original.items():
                        self.assertEqual(saved[key], value)

    def test_decision_stream_is_not_bounded_by_in_memory_trace_limit(self):
        result = run_simulation(**self.options(schedule_trace_limit=0),
                                queue_trace_path=self.directory / "unbounded")
        metadata = result["queue_trace"]
        self.assertEqual(result["schedule_trace"], [])
        self.assertGreater(result["schedule_count"], 0)
        self.assertEqual(metadata["decision_count"], result["schedule_count"])

    def test_early_stop_does_not_pad_future_window(self):
        options = self.options(global_request_max_batch_size=2,
                               tokens_after_reaching_max_bs=1)
        plain = run_simulation(**options)
        traced = run_simulation(**options, queue_trace_path=self.directory / "early",
                                queue_trace_interval_ms=10, queue_trace_end_ms=1000)
        metadata = traced.pop("queue_trace")
        self.assertEqual(json.dumps(plain, sort_keys=True), json.dumps(traced, sort_keys=True))
        self.assertTrue(traced["stopped_early"])
        data = self.arrays(metadata)
        self.assertEqual(data["queue_mean"].shape[0], 1)
        self.assertEqual(data["time_edges_ms"][-1], traced["makespan_ms"])
        self.assertLess(data["time_edges_ms"][-1], 1000)

    def test_ready_queue_integral_matches_completed_ready_waits(self):
        result = run_simulation(**self.options(), queue_trace_path=self.directory / "waits",
                                queue_trace_interval_ms=0.23)
        data = self.arrays(result["queue_trace"])
        area = (data["queue_mean"] * np.diff(data["time_edges_ms"])[:, None, None]).sum(axis=(0, 1))
        for name, observed in (("attention_ready_wait", area[:2].sum()),
                               ("finalize_ready_wait", area[2]),
                               ("expert_ready_wait", area[3:].sum())):
            metric = result["metrics_ms"][name]
            self.assertAlmostEqual(observed, metric["mean"] * metric["count"], places=10)

    def test_execution_error_closes_stream_and_leaves_incomplete_marker(self):
        captured = []

        def recorder(*args, **kwargs):
            instance = QueueTraceRecorder(*args, **kwargs)
            captured.append(instance)
            return instance

        with patch.object(sim_async, "QueueTraceRecorder", side_effect=recorder), \
                patch.object(sim_async.AttentionCostModel, "evaluate", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                run_simulation(**self.options(), queue_trace_path=self.directory / "failed")
        self.assertTrue(captured[0]._decisions.closed)
        metadata = json.loads(Path(captured[0].paths["metadata"]).read_text())
        self.assertFalse(metadata["complete"])
        self.assertFalse(Path(captured[0].paths["npz"]).exists())


if __name__ == "__main__":
    unittest.main()
