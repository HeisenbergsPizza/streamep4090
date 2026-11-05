"""Synthetic integration checks for decode causality and shared GPU resources."""

import csv
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import simpy

from simulation.metrics import Samples
from simulation.sim_async import Branch, GPUWorker, Token, run_simulation


class AsyncIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def profiles(self, *, top_k=2, layers=3, route=None, fixed_expert_ms=None):
        rows = []
        for request_position, rid in enumerate((10, 90, 500)):
            for token_position, token_index in enumerate((7, 40, 200)):
                for layer in range(layers):
                    selected = (route(request_position, token_position, layer) if route
                                else ([0, 2] if top_k == 2 else [(request_position + layer) % 4]))
                    self.assertEqual(len(selected), top_k)
                    row = {"rid": rid, "token_index": token_index, "layer": layer}
                    row.update({f"expert_logical_k{k}": expert for k, expert in enumerate(selected)})
                    rows.append(row)
        trace = self.directory / "synthetic_routing.parquet"
        pq.write_table(pa.Table.from_pylist(rows), trace)
        profile = self.directory / "synthetic_expert.csv"
        with profile.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["batch_size", "avg_time_ms"])
            for size in range(1, 17):
                writer.writerow([size, fixed_expert_ms if fixed_expert_ms is not None else 0.05 + size * 0.01])
        return {"profile_routing_path": str(trace), "expert_profile_path": str(profile)}

    def run_case(self, **overrides):
        options = dict(ep_group_size=2, attn_dp_group_size=2, total_expert_count=4,
                       routing_top_k=2, num_layers=3, total_requests=6, tokens_per_request=3,
                       arrival_rate=0, global_request_max_batch_size=6,
                       max_batch_size=4, max_attention_batch_size=4,
                       attn_service_t=0.2, net_t_attn_to_expert=0.1, net_t_expert_to_attn=0.1,
                       metric_sample_capacity=2, schedule_trace_limit=10000, quiet=True)
        options.update(overrides)
        return run_simulation(**options)

    def assert_shared_compute(self, result, *, layers, experts_per_gpu, batch_cap):
        self.assertFalse(result["schedule_trace_truncated"])
        for gpu in range(len(result["gpu_compute_busy_fraction"])):
            intervals = [entry for entry in result["schedule_trace"] if entry["gpu"] == gpu]
            for before, after in zip(intervals, intervals[1:]):
                self.assertLessEqual(before["end"], after["start"] + 1e-9)
            self.assertLessEqual(result["gpu_compute_busy_fraction"][gpu], 1 + 1e-9)
        for entry in result["schedule_trace"]:
            self.assertGreater(entry["batch_size"], 0)
            self.assertEqual(len(entry["queue_snapshot"]), layers * 2 + 1)
            if entry["stage"] == "expert":
                loads = entry["expert_counts"]
                self.assertEqual(len(loads), experts_per_gpu)
                self.assertEqual(sum(loads), entry["batch_size"])
                self.assertLessEqual(sum(loads), batch_cap)
                self.assertTrue(all(isinstance(count, int) and count >= 0 for count in loads))
                expected = sum((0.05 + count * 0.01) * 10 for count in loads if count)
                self.assertAlmostEqual(entry["end"] - entry["start"], expected)

    def test_all_policies_complete_with_topk_and_multiple_layers(self):
        for policy in ("defrag", "flfs", "mtfs", "fifo"):
            for top_k, layers in ((1, 1), (1, 3), (2, 1), (2, 3)):
                with self.subTest(policy=policy, top_k=top_k, layers=layers):
                    births, completions = {}, {}

                    def begin(token):
                        token.sampled_for_stats = True
                        births[(token.request_id, token.token_index)] = token.birth_time

                    def complete(token, when):
                        completions[(token.request_id, token.token_index)] = when

                    result = self.run_case(
                        **self.profiles(top_k=top_k, layers=layers),
                        scheduling_policy=policy, routing_top_k=top_k, num_layers=layers,
                        per_token_begin_cb=begin, per_token_stats_cb=complete)
                    self.assertEqual(result["completed_tokens"], 18)
                    self.assertEqual(result["completed_requests"], 6)
                    self.assertEqual(len(result["completion_times"]), 18)
                    self.assertFalse(result["stopped_early"])
                    for key in ("inflight_tokens_at_end", "incomplete_joins_at_end",
                                "ready_attention_at_end", "ready_expert_at_end",
                                "inflight_network_branches_at_end", "active_requests_at_end",
                                "waiting_requests_at_end", "unarrived_requests_at_end"):
                        self.assertEqual(result[key], 0, key)
                    for rid in range(6):
                        for index in (1, 2):
                            self.assertEqual(births[(rid, index)], completions[(rid, index - 1)])
                    self.assertTrue(any(size > 1 for size in result["attention_batch_histogram"]))
                    metrics = result["metrics_ms"]
                    expected_counts = {"token_latency": 18, "itl": 12, "request_latency": 6,
                                       "attention_ready_wait": 18 * layers,
                                       "finalize_ready_wait": 18,
                                       "expert_ready_wait": 18 * layers * top_k,
                                       "join_wait": 18 * layers}
                    for name, count in expected_counts.items():
                        self.assertEqual(metrics[name]["count"], count, name)
                        self.assertLessEqual(metrics[name]["quantile_sample_count"], 2)
                        self.assertTrue(metrics[name]["quantiles_are_sampled"])
                        self.assertTrue(math.isfinite(metrics[name]["mean"]))
                        self.assertGreaterEqual(metrics[name]["mean"], 0)
                    self.assert_shared_compute(result, layers=layers, experts_per_gpu=2, batch_cap=4)

    def test_same_gpu_network_is_zero_and_does_not_call_network_model(self):
        def forbidden_network(*args):
            self.fail(f"same-GPU transfer must bypass network model: {args}")

        result = self.run_case(
            **self.profiles(top_k=1, layers=1, route=lambda *args: [0], fixed_expert_ms=0.1),
            ep_group_size=1, attn_dp_group_size=1, total_expert_count=1,
            routing_top_k=1, num_layers=1, total_requests=1, tokens_per_request=1,
            attn_service_t=0.2, net_delay_fn=forbidden_network,
            net_t_attn_to_expert=999, net_t_expert_to_attn=999)
        self.assertAlmostEqual(result["makespan"], 1.2)
        self.assertEqual(result["remote_transfer_count"], 0)
        self.assertEqual(result["metrics_ms"]["network_receive_wait"]["count"], 0)

    def test_attention_profile_uses_initial_context_plus_decode_index(self):
        attention = self.directory / "synthetic_attention.csv"
        with attention.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["batch_size", "context_length", "time_ms"])
            writer.writerows([(1, 100, 0.1), (1, 101, 0.2)])
        result = self.run_case(
            **self.profiles(top_k=1, layers=1, route=lambda *args: [0], fixed_expert_ms=0.1),
            ep_group_size=1, attn_dp_group_size=1, total_expert_count=1,
            routing_top_k=1, num_layers=1, total_requests=1, tokens_per_request=2,
            attention_profile_path=str(attention), initial_context_length=100)
        intervals = [entry for entry in result["schedule_trace"] if entry["stage"] == "attention"]
        self.assertEqual([entry["end"] - entry["start"] for entry in intervals], [1, 2])
        self.assertAlmostEqual(result["makespan"], 5)

    def test_next_decode_waits_for_terminal_finalize(self):
        births, completions = {}, {}

        def begin(token):
            token.sampled_for_stats = True
            births[token.token_index] = token.birth_time

        def complete(token, when):
            completions[token.token_index] = when

        result = self.run_case(
            **self.profiles(top_k=1, layers=1, route=lambda *args: [0], fixed_expert_ms=0.1),
            ep_group_size=1, attn_dp_group_size=1, total_expert_count=1,
            routing_top_k=1, num_layers=1, total_requests=1, tokens_per_request=2,
            attn_service_t=0.2, finalize_per_token_t=3,
            per_token_begin_cb=begin, per_token_stats_cb=complete)
        terminals = [entry for entry in result["schedule_trace"] if entry["stage"] == "finalize"]
        self.assertEqual(len(terminals), 2)
        for index, entry in enumerate(terminals):
            self.assertAlmostEqual(entry["end"] - entry["start"], 3)
            self.assertEqual(completions[index], entry["end"])
        self.assertEqual(births[1], terminals[0]["end"])
        self.assertAlmostEqual(result["makespan"], 8.4)

    def test_fast_destination_starts_before_slow_destination_arrives(self):
        result = self.run_case(
            **self.profiles(top_k=2, layers=1, route=lambda *args: [1, 2], fixed_expert_ms=0.1),
            ep_group_size=3, attn_dp_group_size=3, total_expert_count=3,
            routing_top_k=2, num_layers=1, total_requests=1, tokens_per_request=1,
            attn_service_t=1,
            net_delay_fn=lambda src, dst, copies: 10 if (src, dst) == (0, 2) else 1)
        experts = {entry["gpu"]: entry for entry in result["schedule_trace"] if entry["stage"] == "expert"}
        self.assertAlmostEqual(experts[1]["start"], 2)
        self.assertAlmostEqual(experts[1]["end"], 3)
        self.assertAlmostEqual(experts[2]["start"], 11)
        self.assertLess(experts[1]["end"], experts[2]["start"])
        self.assertAlmostEqual(result["makespan"], 13)
        self.assertAlmostEqual(result["metrics_ms"]["join_wait"]["mean"], 0.9)

    def test_expert_returns_do_not_hold_gpu_compute_resource(self):
        completions = {}

        def begin(token):
            token.sampled_for_stats = True

        def complete(token, when):
            completions[token.request_id] = when

        result = self.run_case(
            **self.profiles(top_k=1, layers=1, route=lambda *args: [1], fixed_expert_ms=0.1),
            total_expert_count=2, routing_top_k=1, num_layers=1,
            total_requests=3, tokens_per_request=1, max_batch_size=1,
            max_attention_batch_size=1, attn_service_t=0,
            net_delay_fn=lambda src, dst, copies: 10 if (src, dst) == (1, 0) else 0,
            per_token_begin_cb=begin, per_token_stats_cb=complete)
        experts = [entry for entry in result["schedule_trace"]
                   if entry["gpu"] == 1 and entry["stage"] == "expert"]
        self.assertEqual(len(experts), 3)
        self.assertEqual([entry["start"] for entry in experts], [0, 1, 2])
        self.assertEqual([entry["end"] for entry in experts], [1, 2, 3])
        # Requests 0 and 2 live on GPU 0. Both require a remote return from GPU 1.
        # By the third expert launch one earlier remote branch has finished;
        # compute continues while its receive transfer is still outstanding.
        self.assertGreater(min(completions[0], completions[2]), experts[-1]["end"])
        self.assertGreaterEqual(min(completions[0], completions[2]), 11)
        self.assertGreater(result["metrics_ms"]["network_receive_wait"]["max"], 0)

    def test_early_stop_reports_unfinished_work_and_bounded_trace(self):
        result = self.run_case(
            **self.profiles(top_k=2, layers=3), total_requests=8,
            global_request_max_batch_size=4, max_attention_batch_size=1,
            tokens_after_reaching_max_bs=1, schedule_trace_limit=2)
        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["completed_tokens"], 1)
        self.assertEqual(result["completed_requests"], 0)
        self.assertGreater(result["inflight_tokens_at_end"], 0)
        self.assertGreater(result["active_requests_at_end"], 0)
        self.assertGreater(result["waiting_requests_at_end"], 0)
        self.assertEqual(result["metrics_ms"]["token_latency"]["count"], 1)
        self.assertEqual(result["metrics_ms"]["request_latency"]["count"], 0)
        self.assertEqual(len(result["schedule_trace"]), 2)
        self.assertTrue(result["schedule_trace_truncated"])
        self.assertTrue(all(0 <= fraction <= 1 + 1e-9 for fraction in result["gpu_compute_busy_fraction"]))

    def test_unknown_policy_fails_before_profile_io(self):
        with self.assertRaisesRegex(ValueError, "Unknown scheduling policy"):
            self.run_case(scheduling_policy="not-a-policy",
                          profile_routing_path="/does/not/exist", expert_profile_path="/does/not/exist")

    def test_expert_batch_cap_preserves_ready_order_with_deterministic_ties(self):
        for policy in ("defrag", "flfs", "mtfs", "fifo"):
            for ready_times, selected in (((5, 4, 0), 2), ((7, 7, 7), 0)):
                with self.subTest(policy=policy, ready_times=ready_times):
                    trace = []
                    env = simpy.Environment(initial_time=10)
                    sim = SimpleNamespace(
                        env=env, num_layers=1, experts_per_gpu=3, top_k=1,
                        policy=policy, lookahead_steps=1, lookback_steps=0, weight_decay=0.8,
                        expert_batch_cap=1, ready_expert=3,
                        stats={"expert_ready_wait": Samples(10)},
                        expert_cost=lambda loads: 1, capture_schedule=trace.append)
                    worker = GPUWorker(sim, rank=0)
                    tokens = [Token(index, 0, index, 0) for index in range(3)]
                    for expert, ready_at in enumerate(ready_times):
                        worker.experts[0][expert].append((Branch(tokens[expert], expert, 0), ready_at))
                    # Run the initial scheduling decision, stopping before its
                    # completion/return path; only one branch fits in the batch.
                    env.step()
                    self.assertEqual(len(trace), 1)
                    self.assertEqual(trace[0]["expert_counts"],
                                     [int(expert == selected) for expert in range(3)])
                    self.assertFalse(worker.experts[0][selected])
                    self.assertEqual(sum(len(queue) for queue in worker.experts[0]), 2)
                    self.assertEqual(tokens[selected].expert_ready_wait, 10 - ready_times[selected])


if __name__ == "__main__":
    unittest.main()
