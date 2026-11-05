"""Steady-window conservation, workload identity, and observation boundaries."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import simpy

from simulation.sim_async import IngressPort, run_simulation
from simulation.steady_metrics import SteadyMetricsObserver


class SteadyMetricsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.routing = self.directory / "routing.parquet"
        rows = [dict(rid=10 + request * 10, token_index=token, layer=0,
                     expert_logical_k0=request % 2)
                for request in range(3) for token in range(3)]
        pq.write_table(pa.Table.from_pylist(rows), self.routing)
        self.profile = self.directory / "expert.csv"
        self.profile.write_text("batch_size,time_ms\n" + "".join(f"{size},1\n" for size in range(1, 9)))

    def options(self, **overrides):
        result = dict(ep_group_size=1, attn_dp_group_size=1, total_expert_count=2,
                      routing_top_k=1, num_layers=1, total_requests=2, tokens_per_request=1,
                      arrival_rate=0, arrival_times_ms=[1, 2], global_request_max_batch_size=1,
                      max_batch_size=8, max_attention_batch_size=8, attn_service_t=2,
                      ticks_per_millisecond=1, rng_seed=72, schedule_trace_limit=10000,
                      expert_profile_path=str(self.profile), profile_routing_path=str(self.routing), quiet=True)
        result.update(overrides)
        return result

    def observer(self, windows=None, **kwargs):
        return SteadyMetricsObserver(windows or [("steady", 0, 10)], sample_interval_ms=2, **kwargs)

    def assert_conserved(self, bucket):
        before, after, counts = bucket["state_at_start"], bucket["state_at_end"], bucket["counts"]
        self.assertEqual(after["outstanding_requests"] - before["outstanding_requests"],
                         counts["arrived_requests"] - counts["completed_requests"])
        self.assertEqual(after["waiting_requests"] - before["waiting_requests"],
                         counts["arrived_requests"] - counts["admitted_requests"])
        self.assertEqual(after["active_requests"] - before["active_requests"],
                         counts["admitted_requests"] - counts["completed_requests"])

    def test_hand_calculated_state_busy_admission_and_request_log(self):
        path = self.directory / "requests.csv"
        result = run_simulation(**self.options(), observer=self.observer(request_log_path=path))
        self.assertEqual(result["makespan_ms"], 7)
        metrics = result["steady_metrics"]
        self.assertEqual(metrics["observation_end_ms"], 10)
        self.assertTrue(metrics["extended_idle_observation"])
        window = metrics["windows"]["steady"]
        self.assertEqual(window["counts"], dict(arrived_requests=2, admitted_requests=2,
                                                completed_requests=2, completed_tokens=2))
        self.assertEqual(window["rates"]["completed_requests_per_second"], 200)
        self.assertAlmostEqual(window["state_mean"]["active_requests"], 0.6)
        self.assertAlmostEqual(window["state_mean"]["waiting_requests"], 0.2)
        self.assertAlmostEqual(window["state_mean"]["outstanding_requests"], 0.8)
        self.assertAlmostEqual(window["state_mean"]["remaining_output_work"], 0.8)
        self.assertEqual(window["gpu_busy_ms"], [6])
        self.assertEqual(window["gpu_busy_fraction"], [0.6])
        self.assertEqual(window["metrics_ms"]["admission_wait"]["sum"], 2)
        self.assertEqual(window["metrics_ms"]["request_latency"]["sum"], 8)
        self.assert_conserved(window)
        for bucket in metrics["timeline"]:
            self.assert_conserved(bucket)
        self.assertEqual(metrics["timeline"][1]["ages_at_end_ms"],
                         {"oldest_unfinished_request": 3, "oldest_waiting_request": 2})
        with path.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual([float(row["arrival_ms"]) for row in records], [1, 2])
        self.assertEqual([float(row["admission_ms"]) for row in records], [1, 4])
        self.assertEqual([float(row["completion_ms"]) for row in records], [4, 7])
        self.assertEqual([row["status"] for row in records], ["completed", "completed"])

    def test_half_open_windows_count_boundaries_and_carry_in_busy(self):
        observer = self.observer([("middle", 2, 4), ("next", 4, 7)])
        result = run_simulation(**self.options(), observer=observer)
        windows = result["steady_metrics"]["windows"]
        middle, following = windows["middle"], windows["next"]
        self.assertEqual(middle["counts"], dict(arrived_requests=1, admitted_requests=0,
                                                completed_requests=0, completed_tokens=0))
        self.assertEqual(following["counts"], dict(arrived_requests=0, admitted_requests=1,
                                                   completed_requests=1, completed_tokens=1))
        self.assertEqual(middle["gpu_busy_ms"], [2])
        self.assertEqual(following["gpu_busy_ms"], [3])
        self.assertEqual(following["metrics_ms"]["request_latency"]["mean"], 3)
        self.assertEqual(following["metrics_ms"]["admission_wait"]["mean"], 2)
        self.assert_conserved(middle)
        self.assert_conserved(following)

    def test_observer_does_not_change_existing_results_for_all_policies(self):
        for policy in ("defrag", "flfs", "mtfs", "fifo"):
            with self.subTest(policy=policy):
                options = self.options(scheduling_policy=policy, total_requests=4,
                                       arrival_times_ms=[0, 0, 1, 7], tokens_per_request=3,
                                       global_request_max_batch_size=2, metric_sample_capacity=2)
                expected = run_simulation(**options)
                actual = run_simulation(**options, observer=self.observer([("all", 0, 100)]))
                observed = actual.pop("steady_metrics")
                self.assertEqual(json.dumps(actual, sort_keys=True), json.dumps(expected, sort_keys=True))
                json.dumps(observed, allow_nan=False)

    def test_explicit_zero_arrivals_match_legacy_zero_rate(self):
        options = self.options(total_requests=4, global_request_max_batch_size=2, tokens_per_request=2)
        legacy = run_simulation(**dict(options, arrival_times_ms=None))
        explicit = run_simulation(**dict(options, arrival_times_ms=[0, 0, 0, 0]))
        self.assertEqual(json.dumps(legacy, sort_keys=True), json.dumps(explicit, sort_keys=True))

    def test_explicit_first_arrival_is_not_forced_to_zero_and_uses_ms(self):
        births = []
        result = run_simulation(**self.options(total_requests=1, arrival_times_ms=[3],
                                ticks_per_millisecond=2, per_token_begin_cb=lambda token: births.append(token.birth_time)))
        self.assertEqual(births, [6])
        self.assertEqual(result["makespan_ms"], 5)

    def test_request_mapping_changes_routes_without_changing_request_identity(self):
        result = run_simulation(**self.options(total_requests=1, arrival_times_ms=[0],
                                               routing_request_indices=[1]), observer=self.observer())
        expert = next(entry for entry in result["schedule_trace"] if entry["stage"] == "expert")
        self.assertEqual(expert["expert_counts"], [0, 1])
        self.assertEqual(result["completed_requests"], 1)
        self.assertEqual(result["steady_metrics"]["request_log_count"], 1)

    def test_invalid_explicit_workloads_are_rejected(self):
        cases = [dict(arrival_times_ms=[1]), dict(arrival_times_ms=[-1, 1]),
                 dict(arrival_times_ms=[2, 1]), dict(arrival_times_ms=[0, float("nan")]),
                 dict(arrival_times_ms=[0, float("inf")]), dict(arrival_times_ms=[False, 1]),
                 dict(routing_request_indices=[0]), dict(routing_request_indices=[0, 3]),
                 dict(routing_request_indices=[0, -1]), dict(routing_request_indices=[0, True]),
                 dict(tokens_per_request=4)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                run_simulation(**self.options(**case))

    def test_empty_explicit_workload_has_known_idle_complete_coverage(self):
        result = run_simulation(**self.options(total_requests=0, arrival_times_ms=[]), observer=self.observer())
        metrics = result["steady_metrics"]
        self.assertEqual(result["makespan_ms"], 0)
        self.assertEqual(result["steps_executed"], 0)
        self.assertEqual(metrics["observation_end_ms"], 10)
        self.assertTrue(metrics["windows"]["steady"]["coverage_complete"])
        self.assertEqual(metrics["windows"]["steady"]["rates"]["output_tokens_per_second"], 0)
        self.assertEqual(metrics["windows"]["steady"]["gpu_busy_ms"], [0])

    def test_early_stop_has_no_idle_padding_and_preserves_point_events_at_window_start(self):
        observer = self.observer([("point", 1, 10), ("future", 5, 10)])
        result = run_simulation(**self.options(tokens_after_reaching_max_bs=0), observer=observer)
        metrics = result["steady_metrics"]
        self.assertTrue(result["stopped_early"])
        self.assertEqual(metrics["observation_end_ms"], 1)
        self.assertFalse(metrics["extended_idle_observation"])
        point = metrics["windows"]["point"]
        self.assertEqual(point["observed_duration_ms"], 0)
        self.assertFalse(point["coverage_complete"])
        self.assertTrue(point["right_closed_at_observation_stop"])
        self.assertEqual(point["counts"]["arrived_requests"], 1)
        self.assertEqual(point["counts"]["admitted_requests"], 1)
        self.assertEqual(point["state_at_end"]["outstanding_requests"], 1)
        self.assertEqual(point["ages_at_end_ms"], {"oldest_unfinished_request": 0,
                                                 "oldest_waiting_request": 0})
        self.assert_conserved(point)
        future = metrics["windows"]["future"]
        self.assertIsNone(future["state_at_start"])
        self.assertIsNone(future["state_at_end"])
        self.assertIsNone(future["rates"]["output_tokens_per_second"])

    def test_early_stop_partial_window_uses_post_stop_state_and_real_duration(self):
        result = run_simulation(**self.options(tokens_after_reaching_max_bs=1), observer=self.observer())
        window = result["steady_metrics"]["windows"]["steady"]
        self.assertEqual(window["observed_duration_ms"], 4)
        self.assertTrue(window["right_closed_at_observation_stop"])
        self.assertFalse(window["coverage_complete"])
        self.assertEqual(window["counts"]["completed_tokens"], 1)
        self.assertEqual(window["state_at_end"]["outstanding_requests"], 1)
        self.assert_conserved(window)

    def test_window_metrics_histograms_and_busy_match_full_run(self):
        result = run_simulation(**self.options(tokens_per_request=3, global_request_max_batch_size=2),
                                observer=self.observer([("all", 0, 100)]))
        window = result["steady_metrics"]["windows"]["all"]
        for name in ("attention_ready_wait", "expert_ready_wait", "finalize_ready_wait", "itl", "admission_wait", "request_latency", "join_wait"):
            old, new = result["metrics_ms"][name], window["metrics_ms"][name]
            self.assertEqual(old["count"], new["count"])
            self.assertAlmostEqual(old["mean"], new["mean"])
            self.assertAlmostEqual(old["max"], new["max"])
        for old_name, name in (("attention_batch_histogram", "attention"),
                               ("expert_group_batch_histogram", "expert_group"),
                               ("per_expert_batch_histogram", "expert_individual"),
                               ("finalize_batch_histogram", "finalize")):
            self.assertEqual(result[old_name], window["batch_histograms"][name])
        self.assertEqual(sum(window["schedule_counts"].values()), result["schedule_count"])
        self.assertEqual(window["gpu_busy_ms"], result["gpu_compute_busy_ticks"])
        for gpu, stages in enumerate(window["gpu_stage_busy_ms"]):
            for stage, duration in stages.items():
                self.assertAlmostEqual(duration, result["gpu_stage_busy_ticks"][gpu].get(stage, 0))
            self.assertAlmostEqual(sum(stages.values()), window["gpu_busy_ms"][gpu])
        self.assertEqual(window["metrics_ms"]["itl"]["count"], 4)

    def test_stage_busy_and_cap_full_are_clipped_to_windows_and_bins(self):
        observer = self.observer([("all", 0, 8), ("clip", 1, 4)])
        result = run_simulation(**self.options(total_requests=1, arrival_times_ms=[0],
                                               finalize_per_token_t=2), observer=observer)
        metrics = result["steady_metrics"]
        self.assertEqual(result["makespan_ms"], 5)
        self.assertEqual(metrics["active_request_cap"], 1)
        whole, clip = metrics["windows"]["all"], metrics["windows"]["clip"]
        self.assertEqual(whole["gpu_stage_busy_ms"], [{"attention": 2, "expert": 1, "finalize": 2}])
        self.assertEqual(clip["gpu_stage_busy_ms"], [{"attention": 1, "expert": 1, "finalize": 1}])
        self.assertEqual(whole["gpu_stage_busy_fraction"],
                         [{"attention": 0.25, "expert": 0.125, "finalize": 0.25}])
        self.assertEqual(whole["active_cap_full_ms"], 5)
        self.assertEqual(whole["active_cap_full_fraction"], 5 / 8)
        self.assertEqual(clip["active_cap_full_fraction"], 1)
        expected = [(2, 0, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0)]
        for bucket, values in zip(metrics["timeline"], expected):
            self.assertEqual(tuple(bucket["gpu_stage_busy_ms"][0].values()), values)
            self.assertAlmostEqual(sum(values), bucket["gpu_busy_ms"][0])
        self.assertEqual(sum(bucket["active_cap_full_ms"] for bucket in metrics["timeline"]), 5)

    def test_fixed_delay_receive_occupancy_is_unmodeled_not_zero(self):
        result = run_simulation(**self.options(ep_group_size=2, attn_dp_group_size=2,
                                               routing_request_indices=[1, 0],
                                               net_t_attn_to_expert=3, net_t_expert_to_attn=4),
                                observer=self.observer([("all", 0, 50)]))
        metrics = result["steady_metrics"]
        self.assertGreater(result["remote_transfer_count"], 0)
        self.assertEqual(metrics["receive_resource_mode"], "fixed_delay_no_contention")
        for bucket in list(metrics["windows"].values()) + metrics["timeline"]:
            self.assertEqual(bucket["receive_resource_mode"], "fixed_delay_no_contention")
            for field in ("receive_busy_ms", "receive_busy_fraction", "receive_queue_message_ms",
                          "receive_queue_messages_mean"):
                self.assertIsNone(bucket[field])

    def test_queue_switches_distinguish_experts_without_changing_stage_or_layer_switches(self):
        # Two dispatches to the same expert, then its sibling, then the next
        # layer, and finally attention. Only the sibling transition differs
        # from the existing layer-level queue-switch metric.
        sequence = [("expert", 0, 0), ("expert", 0, 0),
                    ("expert", 0, 1), ("expert", 1, 1),
                    ("attention", 1, None), ("attention", 1, None)]
        for include_expert_id in (False, True):
            with self.subTest(include_expert_id=include_expert_id):
                observer = self.observer()
                self.resource_fixture(observer, num_gpus=2)
                for start, (stage, layer, expert_id) in enumerate(sequence):
                    trace = dict(start=start, gpu=0, stage=stage, layer=layer,
                                 batch_size=1,
                                 expert_counts=None if stage == "attention"
                                 else [int(index == expert_id) for index in range(2)])
                    if include_expert_id:
                        trace["expert_id"] = expert_id
                    observer.scheduled(trace)
                bucket = observer.windows["steady"]
                self.assertEqual(bucket.queue_switches, [3 if include_expert_id else 2, 0])
                self.assertEqual(bucket.stage_switches, [1, 0])
                self.assertEqual(bucket.layer_switches, [1, 0])

    @staticmethod
    def resource_fixture(observer, num_gpus=4):
        env = simpy.Environment()
        sim = SimpleNamespace(env=env, ticks_per_ms=1, num_gpus=num_gpus, active_cap=1,
                              requests={}, waiting=[], arrived_requests=0, completed_requests=0,
                              ready_attention=0, ready_expert=0, joins={}, inflight_branches=0,
                              tokens_per_request=1, completed_tokens=0,
                              net_delay_fn=lambda *args: 1,
                              workers=[SimpleNamespace(current_compute=None) for _ in range(num_gpus)])
        sim.ports = [IngressPort(env, gpu, 2, observer=observer) for gpu in range(num_gpus)]
        observer.bind(sim)
        return sim

    def test_receive_fifo_busy_queue_and_independent_channels_exact_integrals(self):
        observer = self.observer([("all", 0, 8), ("middle", 2, 5)])
        sim = self.resource_fixture(observer)

        def transfer_at(time, source, delay):
            yield sim.env.timeout(time)
            yield sim.ports[1].transfer(source, delay)

        # Intra services [0,4) and [4,7); its second message waits [1,4).
        # Inter service [.5,2.5) overlaps intra. The final two bypass resources.
        for args in ((0, 0, 4), (1, 0, 3), (0.5, 2, 2), (1.5, 1, 100), (1.5, 2, 0)):
            sim.env.process(transfer_at(*args))
        while sim.env.peek() < float("inf"):
            observer.advance(sim, sim.env.peek())
            sim.env.step()
        metrics = observer.finish(sim, fully_drained=True)
        whole, middle = metrics["windows"]["all"], metrics["windows"]["middle"]
        self.assertEqual(sim.env.now, 7)
        self.assertEqual(whole["receive_busy_ms"], {"intra": [0, 7, 0, 0], "inter": [0, 2, 0, 0]})
        self.assertEqual(whole["receive_busy_fraction"]["intra"][1], 7 / 8)
        self.assertEqual(whole["receive_queue_message_ms"]["intra"], [0, 3, 0, 0])
        self.assertEqual(whole["receive_queue_messages_mean"]["intra"][1], 3 / 8)
        self.assertEqual(whole["receive_queue_message_ms"]["inter"], [0, 0, 0, 0])
        self.assertEqual(whole["metrics_ms"]["network_receive_wait"]["count"], 3)
        self.assertEqual(whole["metrics_ms"]["network_receive_wait"]["sum"], 3)
        self.assertEqual(middle["receive_busy_ms"]["intra"][1], 3)
        self.assertEqual(middle["receive_busy_ms"]["inter"][1], 0.5)
        self.assertEqual(middle["receive_queue_message_ms"]["intra"][1], 2)
        self.assertEqual(middle["receive_queue_messages_mean"]["intra"][1], 2 / 3)
        self.assertEqual([bucket["receive_busy_ms"]["intra"][1] for bucket in metrics["timeline"]],
                         [2, 2, 2, 1])
        self.assertEqual([bucket["receive_queue_message_ms"]["intra"][1] for bucket in metrics["timeline"]],
                         [1, 2, 0, 0])

    def test_early_stop_clips_ongoing_compute_and_receive_services(self):
        observer = self.observer([("all", 0, 10), ("clip", 1, 5), ("future", 5, 8)])
        sim = self.resource_fixture(observer)
        sim.workers[0].current_compute = ("expert", 0, 10)
        sim.requests[0] = (0, 0, 0)
        sim.arrived_requests = 1
        sim.ports[1].transfer(0, 10)
        stop = sim.env.timeout(3)
        while not stop.processed:
            observer.advance(sim, sim.env.peek())
            sim.env.step()
        metrics = observer.finish(sim, fully_drained=False)
        self.assertEqual(metrics["observation_end_ms"], 3)
        for label, duration in (("all", 3), ("clip", 2)):
            bucket = metrics["windows"][label]
            self.assertEqual(bucket["gpu_stage_busy_ms"][0]["expert"], duration)
            self.assertEqual(bucket["gpu_busy_ms"][0], duration)
            self.assertEqual(bucket["receive_busy_ms"]["intra"][1], duration)
            self.assertEqual(bucket["active_cap_full_ms"], duration)
            self.assertEqual(bucket["active_cap_full_fraction"], 1)
        future = metrics["windows"]["future"]
        self.assertIsNone(future["gpu_stage_busy_fraction"][0]["expert"])
        self.assertIsNone(future["receive_busy_fraction"]["intra"][1])
        self.assertIsNone(future["active_cap_full_fraction"])

    def test_contention_observation_preserves_results_and_drain_has_no_busy_padding(self):
        for policy in ("defrag", "flfs"):
            with self.subTest(policy=policy):
                options = self.options(scheduling_policy=policy, ep_group_size=2, attn_dp_group_size=2,
                                       n_gpu_per_host=2, total_requests=6, arrival_times_ms=[0] * 6,
                                       routing_request_indices=[1] * 6, tokens_per_request=3,
                                       global_request_max_batch_size=6, max_batch_size=1,
                                       max_attention_batch_size=1, attn_service_t=0,
                                       net_delay_fn=lambda src, dst, copies: 4, metric_sample_capacity=2)
                expected = run_simulation(**options)
                actual = run_simulation(**options, observer=self.observer([("all", 0, 200)]))
                observed = actual.pop("steady_metrics")
                self.assertEqual(json.dumps(actual, sort_keys=True), json.dumps(expected, sort_keys=True))
                json.dumps(observed, allow_nan=False)
                window = observed["windows"]["all"]
                self.assertGreater(sum(window["receive_queue_message_ms"]["intra"]), 0)
                self.assertEqual(window["receive_busy_ms"]["inter"], [0, 0])
                self.assertAlmostEqual(sum(window["receive_busy_ms"]["intra"]),
                                       actual["remote_transfer_count"] * 4)
                self.assertAlmostEqual(sum(window["receive_queue_message_ms"]["intra"]),
                                       window["metrics_ms"]["network_receive_wait"]["sum"])
                for bucket in observed["timeline"]:
                    for gpu, stages in enumerate(bucket["gpu_stage_busy_ms"]):
                        self.assertAlmostEqual(sum(stages.values()), bucket["gpu_busy_ms"][gpu])
                    if bucket["start_ms"] >= actual["makespan_ms"]:
                        self.assertEqual(bucket["receive_busy_ms"]["intra"], [0, 0])
                        self.assertEqual(bucket["receive_queue_message_ms"]["intra"], [0, 0])
                        self.assertEqual(bucket["gpu_busy_ms"], [0, 0])

    def test_final_event_at_sample_boundary_is_kept_in_explicit_point_bin(self):
        observer = SteadyMetricsObserver([("all", 0, 3)], sample_interval_ms=3)
        result = run_simulation(**self.options(total_requests=1, arrival_times_ms=[0]), observer=observer)
        metrics = result["steady_metrics"]
        self.assertEqual(result["makespan_ms"], 3)
        self.assertEqual(metrics["windows"]["all"]["counts"]["completed_tokens"], 0)
        point = metrics["timeline"][-1]
        self.assertEqual(point["start_ms"], point["end_ms"])
        self.assertTrue(point["right_closed_at_observation_stop"])
        self.assertEqual(point["counts"]["completed_tokens"], 1)
        self.assert_conserved(point)

    def test_invalid_windows_and_reused_observer_are_rejected(self):
        for windows in ([], [("a", 1, 1)], [("a", -1, 1)], [("a", 0, float("inf"))],
                        [("a", 0, 1), ("a", 1, 2)]):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                SteadyMetricsObserver(windows)
        with self.assertRaises(ValueError):
            SteadyMetricsObserver([("a", 0, 1)], sample_interval_ms=0)
        observer = self.observer()
        run_simulation(**self.options(), observer=observer)
        with self.assertRaisesRegex(ValueError, "fresh"):
            run_simulation(**self.options(), observer=observer)

    def test_moments_outside_windows_do_not_consume_wait_generator(self):
        observer = self.observer([("late", 5, 10)])
        sim = SimpleNamespace(ticks_per_ms=1, num_gpus=1, requests={}, waiting=[],
                              arrived_requests=0, completed_requests=0, ready_attention=0, ready_expert=0,
                              joins={}, inflight_branches=0, tokens_per_request=1, completed_tokens=0)
        observer.bind(sim)
        def invalid():
            self.fail("outside-window waits should not be traversed")
            yield 1
        observer.metric(0, "expert_ready_wait", invalid())


if __name__ == "__main__":
    unittest.main()
