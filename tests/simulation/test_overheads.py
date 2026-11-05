"""Hand-computed checks for assumed batch costs and their causal integration.

These fixtures validate the simulator's arithmetic and accounting, not its
accuracy on the paper's GPUs or on a local deployment.
"""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from simulation import sim_async, sim_sync, sim_tbo
from simulation.overheads import BatchOverheadModel
from simulation.steady_metrics import SteadyMetricsObserver


class BatchOverheadCostTests(unittest.TestCase):
    def test_paper_uses_schedule_row_and_its_stage_specific_overlap(self):
        model = BatchOverheadModel(preset="paper", ticks_per_millisecond=10)
        attention = model.evaluate("attention", 2, batch_size=4)
        expert = model.evaluate("expert", 3, batch_size=7)
        # Table 4 Schedule: 127.2 us / 50.6% hidden, 29.6 us / 68.3% hidden.
        # The 5% GPU reformation ratio is an assumption, not a Table 4 value.
        self.assertAlmostEqual(attention.cpu_raw_ticks, 1.272)
        self.assertAlmostEqual(attention.cpu_exposed_ticks, 0.628368)
        self.assertAlmostEqual(expert.cpu_raw_ticks, 0.296)
        self.assertAlmostEqual(expert.cpu_exposed_ticks, 0.093832)
        self.assertAlmostEqual(attention.reformation_raw_ticks, 0.1)
        self.assertAlmostEqual(expert.reformation_raw_ticks, 0.15)
        self.assertAlmostEqual(attention.total_ticks + expert.total_ticks, 5.9722)
        self.assertAlmostEqual(attention.gpu_ticks + expert.gpu_ticks, 5.25)

    def test_microseconds_convert_once_and_ratios_are_unit_invariant(self):
        for units in (1, 10, 1000):
            with self.subTest(ticks_per_millisecond=units):
                model = BatchOverheadModel(preset="paper", ticks_per_millisecond=units)
                estimate = model.evaluate("attention", 2 * units, batch_size=1)
                self.assertAlmostEqual(estimate.base_compute_ticks / units, 2)
                self.assertAlmostEqual(estimate.cpu_raw_ticks / units, 0.1272)
                self.assertAlmostEqual(estimate.cpu_exposed_ticks / units, 0.0628368)
                self.assertAlmostEqual(estimate.reformation_raw_ticks / units, 0.1)
                self.assertAlmostEqual(estimate.total_ticks / units, 2.1628368)

    def test_schedule_charge_is_per_launch_not_per_token(self):
        model = BatchOverheadModel(preset="paper", reformation_compute_ratio=0)
        for stage in ("attention", "expert"):
            with self.subTest(stage=stage):
                one = model.evaluate(stage, 2, batch_size=1)
                many = model.evaluate(stage, 2, batch_size=64)
                self.assertEqual(one.cpu_raw_ticks, many.cpu_raw_ticks)
                self.assertEqual(one.cpu_exposed_ticks, many.cpu_exposed_ticks)
                self.assertEqual(one.total_ticks, many.total_ticks)

    def test_explicit_scale_and_overlap_apply_to_disjoint_components(self):
        model = BatchOverheadModel(
            preset="paper", ticks_per_millisecond=10, cpu_schedule_scale=2,
            cpu_schedule_overlap=0.25, reformation_compute_ratio=0.4,
            reformation_overlap=0.5)
        estimate = model.evaluate("expert", 3, batch_size=5)
        self.assertAlmostEqual(estimate.cpu_raw_ticks, 0.592)
        self.assertAlmostEqual(estimate.cpu_exposed_ticks, 0.444)
        self.assertAlmostEqual(estimate.reformation_raw_ticks, 1.2)
        self.assertAlmostEqual(estimate.reformation_exposed_ticks, 0.6)
        self.assertAlmostEqual(estimate.gpu_ticks, 3.6)
        self.assertAlmostEqual(estimate.total_ticks, 4.044)
        exported = estimate.as_dict()
        for field in ("base_compute_ticks", "cpu_raw_ticks", "cpu_exposed_ticks",
                      "reformation_raw_ticks", "reformation_exposed_ticks",
                      "gpu_ticks", "total_ticks"):
            self.assertAlmostEqual(exported[field], getattr(estimate, field))

    def test_fully_hidden_components_have_no_exposed_cost(self):
        model = BatchOverheadModel(preset="paper", cpu_schedule_overlap=1,
                                   reformation_overlap=1)
        for stage in ("attention", "expert"):
            with self.subTest(stage=stage):
                estimate = model.evaluate(stage, 3, batch_size=1)
                self.assertGreater(estimate.cpu_raw_ticks, 0)
                self.assertGreater(estimate.reformation_raw_ticks, 0)
                self.assertEqual(estimate.cpu_exposed_ticks, 0)
                self.assertEqual(estimate.reformation_exposed_ticks, 0)
                self.assertEqual(estimate.total_ticks, 3)
                self.assertEqual(estimate.gpu_ticks, 3)

    def test_disabled_finalize_and_empty_batches_add_no_overhead(self):
        disabled = BatchOverheadModel(preset="none")
        enabled = BatchOverheadModel(preset="paper")
        cases = [(disabled, "attention", 2, 4), (disabled, "expert", 3, 4),
                 (enabled, "finalize", 4, 2), (enabled, "attention", 0, 0),
                 (enabled, "expert", 0, 0)]
        for model, stage, compute, size in cases:
            with self.subTest(stage=stage, compute=compute, batch_size=size):
                estimate = model.evaluate(stage, compute, batch_size=size)
                self.assertEqual(estimate.total_ticks, compute)
                self.assertEqual(estimate.gpu_ticks, compute)
                for field in ("cpu_raw_ticks", "cpu_exposed_ticks",
                              "reformation_raw_ticks", "reformation_exposed_ticks"):
                    self.assertEqual(getattr(estimate, field), 0)

    def test_invalid_parameters_cannot_silently_create_negative_or_nan_time(self):
        invalid = [dict(preset="unknown")]
        for name in ("ticks_per_millisecond", "cpu_schedule_scale", "reformation_compute_ratio"):
            invalid.extend({name: value} for value in (-1, math.nan, math.inf))
        invalid.append(dict(ticks_per_millisecond=0))
        for name in ("cpu_schedule_overlap", "reformation_overlap"):
            invalid.extend({name: value} for value in (-0.1, 1.1, math.nan, math.inf))
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                BatchOverheadModel(**options)


class BatchOverheadIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.routing = self.directory / "routing.parquet"
        rows = [dict(rid=rid, token_index=token, layer=0, expert_logical_k0=0)
                for rid in range(4) for token in range(3)]
        pq.write_table(pa.Table.from_pylist(rows), self.routing)
        self.profile = self.directory / "expert.csv"
        self.profile.write_text("batch_size,time_ms\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n")

    def options(self, **overrides):
        options = dict(
            ep_group_size=1, attn_dp_group_size=1, total_expert_count=1,
            routing_top_k=1, num_layers=1, total_requests=1, tokens_per_request=1,
            arrival_rate=0, max_batch_size=4, max_attention_batch_size=4,
            global_request_max_batch_size=4, initial_context_length=0,
            expert_profile_path=str(self.profile), profile_routing_path=str(self.routing),
            attn_service_t=0.2, ticks_per_millisecond=10)
        options.update(overrides)
        return options

    def run_async(self, **overrides):
        return sim_async.run_simulation(**self.options(
            quiet=True, schedule_trace_limit=1000, **overrides))

    def test_single_token_matches_hand_calculation_in_all_execution_modes(self):
        # Base attention=.2 ticks, expert=1 tick, local communication=0.
        # CPU = (.1272*.494 + .0296*.317) ms * 10 = .7222 ticks.
        # Reformation = .05 * (.2+1) = .06 ticks. Finalize is unchanged.
        expected = 1.2 + 0.7222 + 0.06 + 0.4
        for mode in ("async", "sync", "tbo"):
            with self.subTest(mode=mode):
                result = self.run_async(execution_mode=mode, finalize_per_token_t=0.4,
                                        overhead_config={"preset": "paper"})
                self.assertAlmostEqual(result["makespan"], expected)
                self.assertAlmostEqual(result["completion_times"][0], expected)
                self.assertEqual(result["completed_requests"], 1)
                self.assertAlmostEqual(result["gpu_compute_busy_ticks"][0], 1.66)

    def test_legacy_baselines_charge_the_same_single_token_work(self):
        for module in (sim_sync, sim_tbo):
            with self.subTest(module=module.__name__), contextlib.redirect_stdout(io.StringIO()):
                result = module.run_simulation(**self.options(
                    finalize_per_token_t=0.4, overhead_config={"preset": "paper"}))
                self.assertAlmostEqual(result["makespan"], 2.3822)
                self.assertAlmostEqual(result["completion_times"][0], 2.3822)

    def test_collective_gpu_intervals_exclude_cpu_but_phase_retains_it(self):
        for mode in ("sync", "tbo"):
            with self.subTest(mode=mode):
                result = self.run_async(execution_mode=mode, overhead_config={"preset": "paper"})
                attention = next(phase for phase in result["collective_phase_trace"]
                                 if phase["phase"] == "attention_compute")
                interval = attention["gpu_intervals"][0]
                self.assertAlmostEqual(attention["start"], 0)
                self.assertAlmostEqual(interval["service_start"], 0)
                self.assertAlmostEqual(interval["start"], 0.628368)
                self.assertAlmostEqual(interval["start_ms"], 0.0628368)
                self.assertAlmostEqual(interval["end"] - interval["start"], 0.21)
                self.assertAlmostEqual(attention["end"], interval["end"])

    def test_explicit_none_preserves_default_timings_metrics_and_schedule(self):
        options = dict(total_requests=3, tokens_per_request=2,
                       arrival_times_ms=[0, 0.02, 0.08], finalize_per_token_t=0.3)
        original = self.run_async(**options)
        disabled = self.run_async(**options, overhead_config={"preset": "none"})
        for field in ("makespan", "completion_times", "schedule_trace", "metrics_ms",
                      "steps_executed", "gpu_compute_busy_ticks", "gpu_stage_busy_ticks",
                      "attention_batch_histogram", "expert_group_batch_histogram"):
            with self.subTest(field=field):
                self.assertEqual(original[field], disabled[field])

    def test_full_overlap_preserves_latency_and_gpu_busy_time(self):
        original = self.run_async(tokens_per_request=2)
        hidden = self.run_async(tokens_per_request=2, overhead_config={
            "preset": "paper", "cpu_schedule_overlap": 1, "reformation_overlap": 1})
        for field in ("makespan", "completion_times", "gpu_compute_busy_ticks",
                      "attention_batch_histogram", "expert_group_batch_histogram"):
            with self.subTest(field=field):
                self.assertEqual(original[field], hidden[field])

    def test_cpu_cost_delays_completion_without_becoming_gpu_busy_time(self):
        result = self.run_async(overhead_config={
            "preset": "paper", "reformation_compute_ratio": 0})
        self.assertAlmostEqual(result["makespan"], 1.9222)
        self.assertAlmostEqual(result["gpu_compute_busy_ticks"][0], 1.2)
        self.assertAlmostEqual(result["gpu_compute_busy_fraction"][0], 1.2 / 1.9222)

    def test_arrival_during_cpu_cost_cannot_join_an_already_selected_batch(self):
        result = self.run_async(total_requests=2, arrival_times_ms=[0, 0.03],
                                overhead_config={"preset": "paper"})
        # Request 1 arrives at .3 ticks, during request 0's .628368-tick
        # scheduling cost. The batch membership was fixed at the decision.
        attention = [entry for entry in result["schedule_trace"]
                     if entry["stage"] == "attention"]
        self.assertEqual([entry["batch_size"] for entry in attention], [1, 1])
        self.assertEqual(result["completed_tokens"], 2)
        self.assertEqual(result["ready_attention_at_end"], 0)
        self.assertEqual(result["incomplete_joins_at_end"], 0)

    def test_observer_excludes_cpu_gaps_in_async_and_shared_collectives(self):
        for mode in ("async", "sync", "tbo"):
            with self.subTest(mode=mode):
                observer = SteadyMetricsObserver(
                    [("cpu_only", 0, 0.03), ("whole", 0, 0.3)], sample_interval_ms=0.01)
                result = self.run_async(execution_mode=mode, observer=observer,
                                        overhead_config={"preset": "paper"})
                windows = result["steady_metrics"]["windows"]
                self.assertEqual(windows["cpu_only"]["gpu_busy_ms"], [0])
                self.assertEqual(windows["cpu_only"]["gpu_busy_fraction"], [0])
                # .2 attention + 1 expert ticks, each with 5% exposed reform.
                self.assertAlmostEqual(windows["whole"]["gpu_busy_ms"][0], 0.126)
                self.assertAlmostEqual(windows["whole"]["gpu_stage_busy_ms"][0]["attention"], 0.021)
                self.assertAlmostEqual(windows["whole"]["gpu_stage_busy_ms"][0]["expert"], 0.105)

    def test_early_stop_inside_another_gpus_cpu_gap_adds_no_gpu_busy_time(self):
        result = self.run_async(
            ep_group_size=2, attn_dp_group_size=2, total_expert_count=2,
            total_requests=2, arrival_times_ms=[0, 0.18],
            global_request_max_batch_size=2, tokens_after_reaching_max_bs=1,
            overhead_config={"preset": "paper"})
        # GPU 0 finishes request 0 at 1.9822 ticks. Request 1 was admitted to
        # GPU 1 at 1.8 ticks and still incurs CPU scheduling until 2.428368.
        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["completed_tokens"], 1)
        self.assertAlmostEqual(result["makespan"], 1.9822)
        self.assertAlmostEqual(result["gpu_compute_busy_ticks"][0], 1.26)
        self.assertEqual(result["gpu_compute_busy_ticks"][1], 0)
        self.assertEqual(result["gpu_compute_busy_fraction"][1], 0)

    def test_cli_overrides_reach_all_modes_and_artifacts_record_assumptions(self):
        from simulation.run import main

        for mode in ("async", "sync", "tbo"):
            with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()):
                output = self.directory / f"cli_{mode}"
                main(["--mode", mode, "--routing-profile", str(self.routing),
                      "--expert-profile", str(self.profile), "--gpus", "1",
                      "--experts", "1", "--top-k", "1", "--layers", "1",
                      "--requests", "1", "--output-tokens", "1",
                      "--active-request-cap", "1", "--attention-batch-cap", "4",
                      "--expert-batch-cap", "4", "--output-dir", str(output),
                      "--overhead-model", "paper", "--cpu-schedule-scale", "2",
                      "--cpu-schedule-overlap", "0.25", "--reformation-compute-ratio", "0.4",
                      "--reformation-overlap", "0.5"])
                label = "async_defrag" if mode == "async" else mode
                artifact = json.loads((output / f"{label}.json").read_text())
                manifest = json.loads((output / "inputs.json").read_text())
                # Base=1.2; exposed CPU=(1.272+.296)*2*.75=2.352;
                # exposed reformation=1.2*.4*.5=.24 ticks.
                self.assertAlmostEqual(artifact["simulation"]["makespan"], 3.792)
                description = artifact["provenance"]["cost_assumptions"]["batch_overhead"]
                self.assertEqual(description, manifest["provenance"]["cost_assumptions"]["batch_overhead"])
                self.assertEqual(description, artifact["simulation"]["batch_overhead_model"])
                self.assertEqual(description["cpu_schedule_scale"], 2)
                self.assertEqual(description["cpu_schedule_overlap"], 0.25)
                self.assertEqual(description["reformation_compute_ratio"], 0.4)
                self.assertEqual(description["reformation_overlap"], 0.5)
                self.assertFalse(description["hardware_calibrated"])
                self.assertIn("assumed", description["reformation_source"])

    def test_cli_rejects_invalid_or_disabled_overrides_before_creating_outputs(self):
        from simulation.run import main

        for index, invalid in enumerate((
            ["--overhead-model", "paper", "--cpu-schedule-overlap", "1.1"],
            ["--overhead-model", "paper", "--reformation-compute-ratio", "nan"],
            ["--overhead-model", "paper", "--reformation-overlap", "-0.1"],
            ["--cpu-schedule-scale", "2"],
        )):
            output = self.directory / f"invalid_{index}"
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--synthetic", "--output-dir", str(output), *invalid])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
