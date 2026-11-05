"""Causal checks for workload-dependent batch costs, not GPU calibration."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from simulation import sim_async, sim_sync, sim_tbo


class StructuralOverheadIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.profile = self.directory / "expert.csv"
        self.profile.write_text(
            "batch_size,time_ms\n" + "".join(f"{size},{size / 10}\n" for size in range(1, 9)))

    def routing(self, *, top_k=2, layers=2, requests=4, tokens=3):
        path = self.directory / f"routing_k{top_k}_l{layers}_r{requests}_t{tokens}.parquet"
        rows = []
        for rid in range(requests):
            for token in range(tokens):
                for layer in range(layers):
                    row = dict(rid=rid, token_index=token, layer=layer)
                    row.update({f"expert_logical_k{k}": k for k in range(top_k)})
                    rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def options(self, **overrides):
        options = dict(
            ep_group_size=1, attn_dp_group_size=1, total_expert_count=2,
            routing_top_k=2, num_layers=2, total_requests=1, tokens_per_request=1,
            arrival_rate=0, max_batch_size=8, max_attention_batch_size=8,
            global_request_max_batch_size=8, initial_context_length=0,
            expert_profile_path=str(self.profile), attn_service_t=0.2,
            ticks_per_millisecond=10, hidden_dim=32, bytes_per_element=2,
            overhead_config={"preset": "structural"})
        options.update(overrides)
        options.setdefault("profile_routing_path", str(self.routing(
            top_k=options["routing_top_k"], layers=options["num_layers"],
            requests=max(options["total_requests"], 1), tokens=options["tokens_per_request"])))
        return options

    def run_async(self, **overrides):
        overrides.setdefault("quiet", True)
        overrides.setdefault("schedule_trace_limit", 1000)
        return sim_async.run_simulation(**self.options(**overrides))

    def assert_drained(self, result, *, requests, tokens):
        self.assertEqual(result["completed_requests"], requests)
        self.assertEqual(result["completed_tokens"], requests * tokens)
        self.assertEqual(result["ready_attention_at_end"], 0)
        self.assertEqual(result["ready_expert_at_end"], 0)
        self.assertEqual(result["incomplete_joins_at_end"], 0)

    def test_topk_join_preserves_input_rows_and_return_buffer_fragments(self):
        # A grouped two-expert result on one GPU returns one contiguous buffer.
        # Splitting the two experts across GPUs returns two distinct buffers.
        # Both cases must read K rows and write one output activation row.
        for mode in ("async", "sync", "tbo"):
            for gpus in (1, 2):
                with self.subTest(mode=mode, gpus=gpus):
                    result = self.run_async(execution_mode=mode, ep_group_size=gpus,
                                            attn_dp_group_size=gpus)
                    attention = next(row for row in result["schedule_trace"]
                                     if row["stage"] == "attention" and row["layer"] == 1)
                    work = attention["batch_workload"]
                    self.assertEqual(attention["batch_size"], 1)
                    self.assertEqual(work["input_rows"], 2)
                    self.assertEqual(work["fragments"], gpus)
                    self.assertEqual(work["read_bytes"], 2 * 32 * 2)
                    self.assertEqual(work["write_bytes"], 32 * 2)
                    self.assertEqual(work["traffic_bytes"], 3 * 32 * 2)
                    expected_us = 3 + (3 * 32 * 2) / (500 * 1000) + 0.05 * gpus
                    self.assertAlmostEqual(attention["batch_overhead"]["reformation_raw_ticks"],
                                           expected_us / 1000 * 10)
                    self.assert_drained(result, requests=1, tokens=1)

    def test_single_contiguous_expert_fragment_skips_gpu_reformation_but_not_cpu(self):
        for mode in ("async", "sync", "tbo"):
            with self.subTest(mode=mode):
                result = self.run_async(execution_mode=mode)
                experts = [row for row in result["schedule_trace"] if row["stage"] == "expert"]
                self.assertEqual(len(experts), 2)
                for row in experts:
                    self.assertEqual(row["batch_size"], 2)
                    self.assertEqual(row["batch_workload"]["input_rows"], 2)
                    self.assertEqual(row["batch_workload"]["fragments"], 1)
                    self.assertTrue(row["batch_workload"]["contiguous"])
                    self.assertEqual(row["batch_overhead"]["reformation_raw_ticks"], 0)
                    self.assertGreater(row["batch_overhead"]["cpu_raw_ticks"], 0)
                    self.assertTrue(row["batch_overhead_components"]["fast_path"])
                    self.assertEqual(row["batch_overhead_components"]["executed_traffic_bytes"], 0)

    def test_expert_batch_combining_two_dispatch_buffers_pays_reformation(self):
        # The synchronous dispatch barrier makes both input buffers available
        # before this batch is selected. It avoids relying on scheduler ties.
        result = self.run_async(execution_mode="sync", ep_group_size=2, attn_dp_group_size=2,
                                routing_top_k=1, num_layers=1, total_requests=2)
        experts = [row for row in result["schedule_trace"] if row["stage"] == "expert"]
        self.assertEqual(len(experts), 1)
        expert = experts[0]
        work = expert["batch_workload"]
        self.assertEqual(expert["batch_size"], 2)
        self.assertEqual(work["input_rows"], 2)
        self.assertEqual(work["fragments"], 2)
        self.assertFalse(work["contiguous"])
        self.assertEqual(work["traffic_bytes"], 4 * 32 * 2)
        self.assertFalse(expert["batch_overhead_components"]["fast_path"])
        self.assertEqual(expert["batch_overhead_components"]["executed_traffic_bytes"],
                         work["traffic_bytes"])
        expected_us = 3 + (4 * 32 * 2) / (500 * 1000) + 0.05 * 2
        self.assertAlmostEqual(expert["batch_overhead"]["reformation_raw_ticks"],
                               expected_us / 1000 * 10)
        self.assert_drained(result, requests=2, tokens=1)

    def test_capped_expert_slices_keep_forward_provenance_and_get_new_return_buffers(self):
        # One forward payload contains both experts' rows. Each cap=1 slice is
        # contiguous. Combining transfers in a collective must retain the two
        # producing batches' separate buffers; message aggregation is no copy.
        for mode in ("async", "sync", "tbo"):
            with self.subTest(mode=mode):
                result = self.run_async(execution_mode=mode, max_batch_size=1)
                experts = [row for row in result["schedule_trace"] if row["stage"] == "expert"]
                self.assertEqual(len(experts), 4)
                for row in experts:
                    self.assertEqual(row["batch_size"], 1)
                    self.assertEqual(row["batch_workload"]["input_rows"], 1)
                    self.assertEqual(row["batch_workload"]["fragments"], 1)
                    self.assertTrue(row["batch_overhead_components"]["fast_path"])
                second_attention = next(row for row in result["schedule_trace"]
                                        if row["stage"] == "attention" and row["layer"] == 1)
                self.assertEqual(second_attention["batch_workload"]["input_rows"], 2)
                self.assertEqual(second_attention["batch_workload"]["fragments"], 2)
                self.assert_drained(result, requests=1, tokens=1)

    def test_collective_dispatch_coalescing_preserves_capped_attention_output_buffers(self):
        # Each microbatch has two tokens and two cap=1 Attention producers.
        # They share one transfer, but the receiving Expert must gather two
        # distinct source buffers instead of obtaining a free contiguous pack.
        for mode, requests, expert_batches in (("sync", 2, 1), ("tbo", 4, 2)):
            with self.subTest(mode=mode):
                result = self.run_async(execution_mode=mode, total_requests=requests,
                                        num_layers=1, routing_top_k=1,
                                        max_attention_batch_size=1)
                dispatches = [phase for phase in result["collective_phase_trace"]
                              if phase["phase"] == "dispatch"]
                self.assertEqual(len(dispatches), expert_batches)
                self.assertTrue(all(phase["transfer_count"] == 1 for phase in dispatches))
                experts = [row for row in result["schedule_trace"] if row["stage"] == "expert"]
                self.assertEqual(len(experts), expert_batches)
                for row in experts:
                    self.assertEqual(row["batch_size"], 2)
                    self.assertEqual(row["batch_workload"]["input_rows"], 2)
                    self.assertEqual(row["batch_workload"]["fragments"], 2)
                    self.assertFalse(row["batch_workload"]["contiguous"])
                    self.assertFalse(row["batch_overhead_components"]["fast_path"])
                    self.assertGreater(row["batch_overhead"]["reformation_raw_ticks"], 0)
                self.assert_drained(result, requests=requests, tokens=1)

    def test_forward_payload_rows_follow_expert_groups_not_router_topk_order(self):
        routing = self.directory / "reverse_topk.parquet"
        pq.write_table(pa.Table.from_pylist([
            dict(rid=0, token_index=0, layer=layer, expert_logical_k0=1, expert_logical_k1=0)
            for layer in range(2)]), routing)
        result = self.run_async(profile_routing_path=str(routing))
        experts = [row for row in result["schedule_trace"] if row["stage"] == "expert"]
        self.assertEqual(len(experts), 2)
        for row in experts:
            self.assertEqual(row["batch_size"], 2)
            self.assertEqual(row["batch_workload"]["fragments"], 1)
            self.assertTrue(row["batch_workload"]["contiguous"])
            self.assertTrue(row["batch_overhead_components"]["fast_path"])
        self.assert_drained(result, requests=1, tokens=1)

    def test_activation_width_changes_byte_cost_without_changing_cpu_work(self):
        narrow = self.run_async(hidden_dim=32)
        wide = self.run_async(hidden_dim=64)
        self.assertEqual(len(narrow["schedule_trace"]), len(wide["schedule_trace"]))
        for left, right in zip(narrow["schedule_trace"], wide["schedule_trace"]):
            if left["stage"] == "finalize":
                continue
            self.assertEqual(left["batch_size"], right["batch_size"])
            self.assertEqual(left["batch_workload"]["input_rows"], right["batch_workload"]["input_rows"])
            self.assertEqual(left["batch_workload"]["fragments"], right["batch_workload"]["fragments"])
            self.assertAlmostEqual(left["batch_overhead"]["cpu_raw_ticks"],
                                   right["batch_overhead"]["cpu_raw_ticks"])
            if left["stage"] == "attention":
                extra_bytes = right["batch_workload"]["traffic_bytes"] - left["batch_workload"]["traffic_bytes"]
                self.assertGreater(extra_bytes, 0)
                self.assertAlmostEqual(
                    right["batch_overhead"]["reformation_raw_ticks"]
                    - left["batch_overhead"]["reformation_raw_ticks"],
                    extra_bytes / (500 * 1000) / 1000 * 10)

    def test_cpu_queue_cost_uses_dispatch_candidate_scan_scope(self):
        # Async scans the A/E layer slots (including empty slots), while the
        # collective controller already names a phase and has one candidate.
        config = dict(preset="structural", cpu_fixed_us=0, cpu_per_row_us=0,
                      cpu_per_fragment_us=0, cpu_per_queue_us=10)
        for mode, candidates in (("async", 5), ("sync", 1), ("tbo", 1)):
            with self.subTest(mode=mode):
                result = self.run_async(execution_mode=mode, overhead_config=config)
                first = result["schedule_trace"][0]
                self.assertEqual(first["batch_workload"]["candidate_queues"], candidates)
                self.assertEqual(first["batch_overhead_components"]["cpu_queues_us"], 10 * candidates)
                self.assertAlmostEqual(first["batch_overhead"]["cpu_raw_ticks"],
                                       10 * candidates / 1000 * 10)

    def test_all_execution_modes_drain_and_cpu_time_is_excluded_from_gpu_busy(self):
        for mode in ("async", "sync", "tbo"):
            with self.subTest(mode=mode):
                result = self.run_async(
                    execution_mode=mode, ep_group_size=2, attn_dp_group_size=2,
                    total_requests=4, tokens_per_request=2)
                self.assert_drained(result, requests=4, tokens=2)
                gpu_busy = [0.0, 0.0]
                cpu_charged = 0.0
                for row in result["schedule_trace"]:
                    cost = row["batch_overhead"]
                    self.assertAlmostEqual(row["gpu_start"] - row["start"],
                                           cost["cpu_exposed_ticks"])
                    self.assertAlmostEqual(row["end"] - row["gpu_start"], cost["gpu_ticks"])
                    self.assertAlmostEqual(row["compute_start"] - row["gpu_start"],
                                           cost["reformation_exposed_ticks"])
                    gpu_busy[row["gpu"]] += cost["gpu_ticks"]
                    cpu_charged += cost["cpu_exposed_ticks"]
                self.assertGreater(cpu_charged, 0)
                for gpu, expected in enumerate(gpu_busy):
                    self.assertAlmostEqual(result["gpu_compute_busy_ticks"][gpu], expected)

    def test_schedule_trace_capture_cannot_change_timing_or_batch_membership(self):
        for mode in ("async", "sync", "tbo"):
            with self.subTest(mode=mode):
                options = dict(execution_mode=mode, ep_group_size=2, attn_dp_group_size=2,
                               total_requests=4, tokens_per_request=2)
                captured = self.run_async(**options)
                unobserved = self.run_async(**options, schedule_trace_limit=0)
                self.assertTrue(captured["schedule_trace"])
                self.assertEqual(unobserved["schedule_trace"], [])
                for field in ("makespan", "completion_times", "gpu_compute_busy_ticks",
                              "attention_batch_histogram", "expert_group_batch_histogram",
                              "batch_overhead_accounting"):
                    self.assertEqual(captured[field], unobserved[field], field)
                for observed, hidden in zip(captured["batch_overhead_accounting"]["by_gpu"],
                                            unobserved["batch_overhead_accounting"]["by_gpu"]):
                    self.assertTrue(observed["by_stage_workload"])
                    self.assertEqual(observed["by_stage_workload"], hidden["by_stage_workload"])

    def test_legacy_baselines_reject_structural_without_fragment_metadata(self):
        for module in (sim_sync, sim_tbo):
            with self.subTest(module=module.__name__), contextlib.redirect_stdout(io.StringIO()):
                options = self.options()
                options.pop("hidden_dim")
                options.pop("bytes_per_element")
                with self.assertRaisesRegex(ValueError, "structural"):
                    module.run_simulation(**options)


if __name__ == "__main__":
    unittest.main()
