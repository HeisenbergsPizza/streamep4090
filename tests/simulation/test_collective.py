"""Shared-clock synchronous EP and two-microbatch overlap invariants."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from simulation.sim_async import run_simulation
from simulation.steady_metrics import SteadyMetricsObserver


class CollectiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.routing = self.directory / "routing.parquet"
        rows = [dict(rid=request, token_index=token, layer=layer,
                     expert_logical_k0=0, expert_logical_k1=2)
                for request in range(10) for token in range(3) for layer in range(2)]
        pq.write_table(pa.Table.from_pylist(rows), self.routing)
        self.profile = self.directory / "expert.csv"
        self.profile.write_text("batch_size,time_ms\n" + "".join(f"{size},1\n" for size in range(1, 33)))

    def options(self, **overrides):
        options = dict(ep_group_size=2, attn_dp_group_size=2, total_expert_count=4,
                       routing_top_k=2, num_layers=2, total_requests=8, tokens_per_request=2,
                       arrival_rate=0, global_request_max_batch_size=8,
                       max_batch_size=3, max_attention_batch_size=2,
                       attn_service_t=1, finalize_per_token_t=0.1,
                       net_delay_fn=lambda src, dst, copies: 3,
                       ticks_per_millisecond=1, schedule_trace_limit=10000,
                       expert_profile_path=str(self.profile), profile_routing_path=str(self.routing), quiet=True)
        options.update(overrides)
        return options

    def test_conservation_caps_and_shared_compute(self):
        for mode in ("sync", "tbo"):
            with self.subTest(mode=mode):
                result = run_simulation(**self.options(execution_mode=mode))
                self.assertEqual(result["completed_tokens"], 16)
                self.assertEqual(result["completed_requests"], 8)
                self.assertFalse(result["stopped_early"])
                for key in ("ready_attention_at_end", "ready_expert_at_end", "incomplete_joins_at_end",
                            "inflight_tokens_at_end", "inflight_network_branches_at_end",
                            "active_requests_at_end", "waiting_requests_at_end", "unarrived_requests_at_end"):
                    self.assertEqual(result[key], 0, key)
                events = result["schedule_trace"]
                self.assertFalse(result["schedule_trace_truncated"])
                self.assertEqual(sum(e["batch_size"] for e in events if e["stage"] == "attention"), 32)
                self.assertEqual(sum(e["batch_size"] for e in events if e["stage"] == "expert"), 64)
                self.assertEqual(sum(e["batch_size"] for e in events if e["stage"] == "finalize"), 16)
                for event in events:
                    cap = 3 if event["stage"] == "expert" else 2
                    self.assertEqual(event["batch_size"], min(cap, event["eligible_queue_size"]))
                    self.assertEqual(event["scores"], [])
                    self.assertIsNone(event["selected_queue_index"])
                    self.assertEqual(len(event["queue_snapshot"]), 5)
                    if event["stage"] == "expert":
                        self.assertEqual(sum(event["expert_counts"]), event["batch_size"])
                for gpu in range(2):
                    intervals = [e for e in events if e["gpu"] == gpu]
                    for before, after in zip(intervals, intervals[1:]):
                        self.assertLessEqual(before["end"], after["start"] + 1e-10)
                    self.assertAlmostEqual(result["gpu_compute_busy_ticks"][gpu],
                                           sum(e["end"] - e["start"] for e in intervals))

    def test_global_compute_and_microbatch_phase_barriers(self):
        for mode in ("sync", "tbo"):
            result = run_simulation(**self.options(execution_mode=mode))
            phases = result["collective_phase_trace"]
            compute = sorted((p for p in phases if p["resource"] == "global_compute_phase"), key=lambda p: p["start"])
            for before, after in zip(compute, compute[1:]):
                self.assertLessEqual(before["end"], after["start"] + 1e-10)
            for iteration, mb in {(p["iteration"], p["microbatch"]) for p in phases}:
                sequence = [p for p in phases if (p["iteration"], p["microbatch"]) == (iteration, mb)]
                expected = [(layer, phase) for layer in range(2)
                            for phase in ("attention_compute", "dispatch", "expert_compute", "combine")]
                expected.append((2, "finalize_compute"))
                self.assertEqual([(p["layer"], p["phase"]) for p in sequence], expected)
                for before, after in zip(sequence, sequence[1:]):
                    self.assertLessEqual(before["end"], after["start"] + 1e-10)
            boundary = max(p["end"] for p in phases if p["iteration"] == 0)
            self.assertGreaterEqual(min(p["start"] for p in phases if p["iteration"] == 1), boundary)

    def test_two_batch_split_is_balanced_within_each_home_gpu(self):
        result = run_simulation(**self.options(execution_mode="tbo", total_requests=10,
                                               global_request_max_batch_size=10, tokens_per_request=1))
        starts = [p for p in result["collective_phase_trace"]
                  if p["phase"] == "attention_compute" and p["layer"] == 0]
        self.assertEqual([p["token_counts_by_home"] for p in starts], [[3, 3], [2, 2]])

    def test_tbo_overlaps_communication_with_other_microbatch_compute(self):
        for mode in ("sync", "tbo"):
            result = run_simulation(**self.options(execution_mode=mode))
            phases = result["collective_phase_trace"]
            overlap = [(a, b) for a in phases for b in phases
                       if a["resource"] == "receive_ports" and b["resource"] == "global_compute_phase"
                       and min(a["end"], b["end"]) > max(a["start"], b["start"]) + 1e-10]
            self.assertEqual(bool(overlap), mode == "tbo")
            self.assertTrue(all(a["microbatch"] != b["microbatch"] for a, b in overlap))

    def test_next_token_birth_and_compute_iteration_gate(self):
        births, completed = {}, {}

        def begin(token):
            token.sampled_for_stats = True
            births[(token.request_id, token.token_index)] = token.birth_time

        def complete(token, when):
            completed[(token.request_id, token.token_index)] = when

        result = run_simulation(**self.options(execution_mode="tbo", total_requests=6,
                                               per_token_begin_cb=begin, per_token_stats_cb=complete))
        for rid in range(6):
            self.assertEqual(births[(rid, 1)], completed[(rid, 0)])
        first_iteration_end = max(completed[(rid, 0)] for rid in range(6))
        second_start = min(e["start"] for e in result["schedule_trace"] if e["iteration"] == 1)
        self.assertGreaterEqual(second_start, first_iteration_end)
        self.assertTrue(any(births[(rid, 1)] < second_start for rid in range(6)))
        itls = [completed[(rid, 1)] - completed[(rid, 0)] for rid in range(6)]
        self.assertAlmostEqual(result["metrics_ms"]["itl"]["mean"], sum(itls) / 6)

    def test_arrivals_inside_iteration_wait_for_boundary(self):
        births = {}
        result = run_simulation(**self.options(execution_mode="sync", total_requests=3,
                                               arrival_times_ms=[0, 0, 0.5], tokens_per_request=1,
                                               per_token_begin_cb=lambda t: births.setdefault(t.request_id, t.birth_time)))
        end = max(p["end"] for p in result["collective_phase_trace"] if p["iteration"] == 0)
        self.assertEqual(births[0], 0)
        self.assertEqual(births[1], 0)
        self.assertEqual(births[2], end)
        self.assertGreater(end, 0.5)

    def test_expert_cap_merges_ready_heads_by_fifo(self):
        rows = [dict(rid=request, token_index=0, layer=0,
                     expert_logical_k0=1 if request == 0 else 0,
                     expert_logical_k1=3 if request == 0 else 2) for request in range(2)]
        pq.write_table(pa.Table.from_pylist(rows), self.routing)
        result = run_simulation(**self.options(execution_mode="sync", num_layers=1,
                                               total_requests=2, tokens_per_request=1, max_batch_size=1))
        first = next(e for e in result["schedule_trace"] if e["stage"] == "expert" and e["gpu"] == 0)
        self.assertEqual(first["expert_counts"], [0, 1])

    def test_observer_is_transparent_and_conserves_windows(self):
        for mode in ("sync", "tbo"):
            options = self.options(execution_mode=mode)
            plain = run_simulation(**options)
            observer = SteadyMetricsObserver([("all", 0, 200)], sample_interval_ms=5)
            observed = run_simulation(**options, observer=observer)
            metrics = observed.pop("steady_metrics")
            self.assertEqual(json.dumps(plain, sort_keys=True), json.dumps(observed, sort_keys=True))
            window = metrics["windows"]["all"]
            self.assertEqual(window["counts"]["completed_tokens"], 16)
            for bucket in [window] + metrics["timeline"]:
                start, end, counts = bucket["state_at_start"], bucket["state_at_end"], bucket["counts"]
                self.assertEqual(end["outstanding_requests"] - start["outstanding_requests"],
                                 counts["arrived_requests"] - counts["completed_requests"])
            self.assertTrue(all(0 <= x <= 1 for x in window["gpu_busy_fraction"]))

    def test_window_trace_retains_carry_in_and_gated_occupancy_definition(self):
        prefix = self.directory / "trace"
        result = run_simulation(**self.options(execution_mode="tbo", schedule_trace_limit=0,
                                               queue_trace_path=prefix, queue_trace_start_ms=0.1,
                                               queue_trace_end_ms=0.2, queue_trace_interval_ms=0.05))
        metadata = result["queue_trace"]
        self.assertIn("barrier-gated", metadata["queue_definition"])
        with gzip.open(metadata["paths"]["decisions"], "rt") as handle:
            decisions = [json.loads(line) for line in handle]
        with gzip.open(result["collective_phase_trace_path"], "rt") as handle:
            phases = [json.loads(line) for line in handle]
        self.assertTrue(decisions and phases)
        self.assertTrue(all(e["start_ms"] == 0 and e["end_ms"] > 0.2 for e in decisions))
        self.assertTrue(all("queue_snapshot" in e for e in decisions))
        self.assertEqual(phases[0]["phase"], "attention_compute")
        self.assertEqual(phases[0]["start_ms"], 0)
        self.assertGreater(phases[0]["end_ms"], 0.2)

    def test_empty_workload_and_invalid_mode(self):
        for mode in ("sync", "tbo"):
            result = run_simulation(**self.options(execution_mode=mode, total_requests=0))
            self.assertEqual(result["completed_tokens"], 0)
            self.assertEqual(result["makespan_ms"], 0)
            self.assertEqual(result["collective_phase_trace"], [])
            with self.assertRaisesRegex(ValueError, "require expert_scheduling"):
                run_simulation(**self.options(execution_mode=mode, expert_scheduling="expert"))
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            run_simulation(**self.options(execution_mode="invalid"))

    def test_explicit_async_mode_preserves_default_results(self):
        default = run_simulation(**self.options())
        explicit = run_simulation(**self.options(execution_mode="async"))
        self.assertEqual(json.dumps(default, sort_keys=True), json.dumps(explicit, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
