"""Execution invariants for independently scheduled local expert queues."""

import csv
import gzip
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import simpy

from simulation.metrics import Samples
from simulation.sim_async import Branch, GPUWorker, Token, run_simulation


class ExpertWiseAsyncTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def profiles(self, *, top_k=2, layers=3, route=None, fixed_expert_ms=None):
        rows = []
        for request in range(3):
            for token in range(3):
                for layer in range(layers):
                    selected = (route(request, token, layer) if route is not None else
                                ([request % 2, 2 + (request + layer) % 2]
                                 if top_k == 2 else [(request + layer) % 4]))
                    row = {"rid": request, "token_index": token, "layer": layer}
                    row.update({f"expert_logical_k{k}": expert
                                for k, expert in enumerate(selected)})
                    rows.append(row)
        trace = self.directory / "routing.parquet"
        pq.write_table(pa.Table.from_pylist(rows), trace)
        profile = self.directory / "expert.csv"
        with profile.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["batch_size", "avg_time_ms"])
            for size in range(1, 17):
                writer.writerow([size, fixed_expert_ms if fixed_expert_ms is not None
                                 else 0.05 + size * 0.01])
        return {"profile_routing_path": str(trace), "expert_profile_path": str(profile)}

    def run_case(self, **overrides):
        options = dict(ep_group_size=2, attn_dp_group_size=2, total_expert_count=4,
                       routing_top_k=2, num_layers=3, total_requests=6,
                       tokens_per_request=3, arrival_rate=0,
                       global_request_max_batch_size=6,
                       max_batch_size=4, max_attention_batch_size=4,
                       attn_service_t=0.2, net_t_attn_to_expert=0.1,
                       net_t_expert_to_attn=0.1, expert_scheduling="expert",
                       schedule_trace_limit=10000, quiet=True)
        options.update(overrides)
        return run_simulation(**options)

    def worker(self, counts, *, policy="mtfs", cap=2, ready_times=None, rank=0):
        trace, cost_calls = [], []
        env = simpy.Environment(initial_time=10)

        def cost(loads):
            cost_calls.append(list(loads))
            return 1.0

        sim = SimpleNamespace(
            env=env, num_layers=len(counts), experts_per_gpu=len(counts[0]),
            top_k=1, policy=policy, lookahead_steps=1, lookback_steps=0,
            weight_decay=0.8, expert_scheduling="expert", expert_batch_cap=cap,
            ready_expert=sum(map(sum, counts)),
            stats={"expert_ready_wait": Samples(100)},
            expert_cost=cost, capture_schedule=trace.append)
        worker = GPUWorker(sim, rank)
        tid = 0
        for layer, row in enumerate(counts):
            for expert, count in enumerate(row):
                for offset in range(count):
                    token = Token(tid, 0, tid, 0)
                    ready_at = ready_times[layer][expert] if ready_times is not None else offset
                    global_expert = rank * len(row) + expert
                    worker.experts[layer][expert].append(
                        (Branch(token, global_expert, layer), ready_at))
                    tid += 1
        return env, sim, worker, trace, cost_calls

    def test_global_expert_score_wins_over_larger_layer_aggregate(self):
        # Layer 0 has 12 branches in total, but no individual queue can beat
        # layer 1's queue of 7. A layer-first selection would choose incorrectly.
        for policy in ("defrag", "mtfs"):
            with self.subTest(policy=policy):
                env, sim, worker, trace, costs = self.worker(
                    [[4, 4, 4], [0, 0, 7]], policy=policy, rank=1)
                env.step()
                self.assertEqual(len(trace), 1)
                entry = trace[0]
                self.assertEqual((entry["stage"], entry["layer"], entry["local_expert_id"]),
                                 ("expert", 1, 2))
                self.assertEqual(entry["expert_id"], 5)
                self.assertEqual(entry["selected_queue_index"], 8)
                self.assertEqual(entry["queue_snapshot"], [0, 0, 0, 4, 4, 4, 0, 0, 7])
                self.assertEqual(entry["scores"][8], max(entry["scores"]))
                self.assertEqual(costs, [[0, 0, 2]])
                self.assertEqual([len(queue) for queue in worker.experts[0]], [4, 4, 4])
                self.assertEqual([len(queue) for queue in worker.experts[1]], [0, 0, 5])
                self.assertEqual(sim.ready_expert, 17)

    def test_selected_queue_partial_drain_is_fifo_and_other_queues_unchanged(self):
        env, sim, worker, trace, costs = self.worker([[5, 4, 1]], cap=2)
        before = [list(queue) for queue in worker.experts[0]]
        env.step()
        self.assertEqual(costs, [[2, 0, 0]])
        self.assertEqual(list(worker.experts[0][0]), before[0][2:])
        self.assertEqual(list(worker.experts[0][1]), before[1])
        self.assertEqual(list(worker.experts[0][2]), before[2])
        self.assertEqual([item[0].token.expert_ready_wait for item in before[0]],
                         [10, 9, 0, 0, 0])
        self.assertEqual(sim.ready_expert, 8)
        self.assertEqual(trace[0]["batch_size"], 2)
        # Existing occupancy consumers continue to see layer aggregates.
        self.assertEqual(worker.snapshot()[:2], ([0, 0], [8]))

    def test_fifo_uses_each_expert_oldest_ready_time(self):
        env, _, worker, trace, costs = self.worker(
            [[4, 5], [1, 2]], policy="fifo", cap=3,
            ready_times=[[8, 7], [6, 0]])
        env.step()
        self.assertEqual((trace[0]["layer"], trace[0]["local_expert_id"]), (1, 1))
        self.assertEqual(costs, [[0, 2]])
        self.assertEqual([len(queue) for queue in worker.experts[0]], [4, 5])
        self.assertEqual([len(queue) for queue in worker.experts[1]], [1, 0])

    def test_all_policies_preserve_completion_and_shared_gpu_compute(self):
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
                        routing_top_k=top_k, num_layers=layers, scheduling_policy=policy,
                        per_token_begin_cb=begin, per_token_stats_cb=complete)
                    self.assertEqual(result["completed_tokens"], 18)
                    self.assertEqual(result["completed_requests"], 6)
                    self.assertEqual(len(result["completion_times"]), 18)
                    self.assertFalse(result["stopped_early"])
                    self.assertFalse(result["schedule_trace_truncated"])
                    for key in ("inflight_tokens_at_end", "incomplete_joins_at_end",
                                "ready_attention_at_end", "ready_expert_at_end",
                                "inflight_network_branches_at_end", "active_requests_at_end",
                                "waiting_requests_at_end", "unarrived_requests_at_end"):
                        self.assertEqual(result[key], 0, key)
                    for rid in range(6):
                        for token in (1, 2):
                            self.assertEqual(births[(rid, token)], completions[(rid, token - 1)])
                    self.assertEqual(result["metrics_ms"]["join_wait"]["count"], 18 * layers)
                    self.assertEqual(result["metrics_ms"]["expert_ready_wait"]["count"],
                                     18 * layers * top_k)
                    for gpu in range(2):
                        intervals = [entry for entry in result["schedule_trace"]
                                     if entry["gpu"] == gpu]
                        for before, after in zip(intervals, intervals[1:]):
                            self.assertLessEqual(before["end"], after["start"] + 1e-9)
                        self.assertLessEqual(result["gpu_compute_busy_fraction"][gpu], 1 + 1e-9)
                    for entry in result["schedule_trace"]:
                        self.assertEqual(entry["expert_scheduling"], "expert")
                        self.assertEqual(len(entry["queue_snapshot"]), layers + 1 + layers * 2)
                        self.assertEqual(len(entry["scores"]), len(entry["queue_snapshot"]))
                        self.assertTrue(all(math.isfinite(score) for score in entry["scores"]))
                        selected = entry["selected_queue_index"]
                        self.assertGreater(entry["queue_snapshot"][selected], 0)
                        candidates = [index for index, count in enumerate(entry["queue_snapshot"]) if count]
                        self.assertEqual(entry["scores"][selected],
                                         max(entry["scores"][index] for index in candidates))
                        if entry["stage"] == "expert":
                            loads, expert = entry["expert_counts"], entry["local_expert_id"]
                            self.assertEqual(sum(count > 0 for count in loads), 1)
                            self.assertEqual(loads[expert], entry["batch_size"])
                            self.assertLessEqual(entry["batch_size"], 4)
                            self.assertEqual(entry["expert_id"], entry["gpu"] * 2 + expert)
                            self.assertEqual(selected, layers + 1 + entry["layer"] * 2 + expert)
                            self.assertAlmostEqual(entry["end"] - entry["start"],
                                                   (0.05 + 0.01 * entry["batch_size"]) * 10)
                        else:
                            self.assertIsNone(entry["local_expert_id"])
                            self.assertIsNone(entry["expert_id"])
                            self.assertEqual(selected, entry["layer"])

    def test_topk_join_waits_for_remote_expert_before_next_layer(self):
        result = self.run_case(
            **self.profiles(layers=2, route=lambda *args: [1, 2], fixed_expert_ms=0.1),
            num_layers=2, total_requests=1, tokens_per_request=1, attn_service_t=2,
            net_delay_fn=lambda src, dst, count: 10 if (src, dst) == (0, 1) else 3)
        trace = result["schedule_trace"]
        for layer in range(2):
            attention = next(entry for entry in trace
                             if entry["stage"] == "attention" and entry["layer"] == layer)
            experts = [entry for entry in trace
                       if entry["stage"] == "expert" and entry["layer"] == layer]
            self.assertEqual({entry["expert_id"] for entry in experts}, {1, 2})
            local = next(entry for entry in experts if entry["gpu"] == 0)
            remote = next(entry for entry in experts if entry["gpu"] == 1)
            self.assertAlmostEqual(local["start"], attention["end"])
            self.assertAlmostEqual(remote["start"], attention["end"] + 10)
            next_stage = next(entry for entry in trace
                              if entry["stage"] != "expert" and entry["layer"] == layer + 1)
            self.assertAlmostEqual(next_stage["start"], remote["end"] + 3)
            self.assertLess(local["end"], remote["start"])
        self.assertAlmostEqual(result["makespan"], 32)
        self.assertEqual(result["incomplete_joins_at_end"], 0)

    def test_tracing_changes_no_execution_and_records_expert_layout(self):
        profiles = self.profiles()
        without = self.run_case(**profiles, schedule_trace_limit=0)
        with_trace = self.run_case(**profiles, queue_trace_path=self.directory / "queues")
        for key in without:
            if key not in ("schedule_trace", "schedule_trace_truncated"):
                self.assertEqual(without[key], with_trace[key], key)
        metadata = with_trace["queue_trace"]
        self.assertEqual(metadata["occupancy_granularity"], "layer_aggregate")
        self.assertEqual(metadata["decision_queue_layout"]["expert_scheduling"], "expert")
        with np.load(metadata["paths"]["npz"], allow_pickle=False) as data:
            self.assertEqual(data["queue_mean"].shape[1:], (2, 7))
            self.assertTrue(np.all(data["queue_mean"] >= 0))
        with gzip.open(metadata["paths"]["decisions"], "rt") as file:
            decisions = [json.loads(line) for line in file]
        self.assertEqual(len(decisions), with_trace["schedule_count"])
        for disk, memory in zip(decisions, with_trace["schedule_trace"]):
            self.assertEqual({key: disk[key] for key in memory}, memory)
            self.assertEqual(len(disk["queue_snapshot"]), 10)

    def grouped_profile(self, rows):
        path = self.directory / "grouped.csv"
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["expert_counts", "time_ms"])
            writer.writerows((json.dumps(counts), time) for counts, time in rows)
        return str(path)

    def test_exact_grouped_profile_receives_selected_onehot_loads(self):
        result = self.run_case(
            **self.profiles(layers=1, route=lambda *args: [0, 1]),
            ep_group_size=1, attn_dp_group_size=1, total_expert_count=2,
            num_layers=1, total_requests=1, tokens_per_request=1,
            grouped_profile_path=self.grouped_profile([([1, 0], 0.3), ([0, 1], 0.7)]),
            expert_cost_scale=2)
        experts = [entry for entry in result["schedule_trace"] if entry["stage"] == "expert"]
        self.assertEqual([entry["expert_counts"] for entry in experts], [[1, 0], [0, 1]])
        self.assertEqual([entry["end"] - entry["start"] for entry in experts], [6, 14])
        self.assertAlmostEqual(result["makespan"], 20.2)
        self.assertEqual(result["cost_model"]["expert"], "exact_grouped_profile")

    def test_missing_onehot_grouped_measurement_does_not_fall_back(self):
        with self.assertRaisesRegex(ValueError, "no exact measurement for expert_counts"):
            self.run_case(
                **self.profiles(layers=1, route=lambda *args: [0, 1]),
                ep_group_size=1, attn_dp_group_size=1, total_expert_count=2,
                num_layers=1, total_requests=1, tokens_per_request=1,
                grouped_profile_path=self.grouped_profile([([1, 1], 0.3)]))


if __name__ == "__main__":
    unittest.main()
