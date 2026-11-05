"""Structural cost arithmetic/validation; coefficients are assumptions, not profiles."""

import argparse
from dataclasses import FrozenInstanceError, replace
import json
import math
import unittest

from simulation.overheads import (
    BatchOverheadModel,
    BatchWorkload,
    STRUCTURAL_DEFAULTS,
    add_overhead_arguments,
    overhead_config_from_args,
)


class StructuralCostTests(unittest.TestCase):
    def setUp(self):
        self.workload = BatchWorkload(
            input_rows=128, fragments=4, candidate_queues=8,
            hidden_dim=2048, bytes_per_element=2)

    def test_default_topk_pack_accounts_for_all_input_and_output_bytes(self):
        estimate = BatchOverheadModel(preset="structural").evaluate(
            "attention", 2, batch_size=32, workload=self.workload)
        # K=4: read 128 rows and write 32 rows, each 4096 bytes.
        # CPU = 10 + .02*128 + .1*4 + .05*8 = 13.36 us.
        # GPU = 3 + 655360/(500*1000) + .05*4 = 4.51072 us.
        self.assertAlmostEqual(estimate.cpu_raw_ticks, 0.1336)
        self.assertAlmostEqual(estimate.cpu_exposed_ticks, 0.1336)
        self.assertAlmostEqual(estimate.reformation_raw_ticks, 0.0451072)
        self.assertAlmostEqual(estimate.reformation_exposed_ticks, 0.0451072)
        self.assertAlmostEqual(estimate.total_ticks, 2.1787072)
        self.assertEqual(estimate.details["workload"]["read_bytes"], 524288)
        self.assertEqual(estimate.details["workload"]["write_bytes"], 131072)
        self.assertEqual(estimate.details["workload"]["traffic_bytes"], 655360)
        self.assertEqual(estimate.details["components"]["executed_traffic_bytes"], 655360)
        self.assertFalse(estimate.details["components"]["fast_path"])
        self.assertAlmostEqual(estimate.details["components"]["cpu_unscaled_us"], 13.36)
        self.assertAlmostEqual(estimate.details["components"]["reformation_raw_us"], 4.51072)
        self.assertNotIn("details", estimate.as_dict())
        self.assertTrue(all(isinstance(value, (float, int)) for value in estimate.as_dict().values()))
        json.dumps(estimate.details, allow_nan=False)

    def test_byte_and_fragment_costs_do_not_scale_with_compute_time(self):
        model = BatchOverheadModel(preset="structural")
        short = model.evaluate("expert", 0.1, 128, self.workload)
        long = model.evaluate("expert", 100, 128, self.workload)
        self.assertEqual(short.cpu_raw_ticks, long.cpu_raw_ticks)
        self.assertEqual(short.reformation_raw_ticks, long.reformation_raw_ticks)

    def test_more_fragments_add_cpu_and_gpu_cost_at_fixed_data_size(self):
        model = BatchOverheadModel(preset="structural")
        one = model.evaluate("expert", 1, 128, replace(self.workload, fragments=1))
        four = model.evaluate("expert", 1, 128, self.workload)
        self.assertAlmostEqual(four.cpu_raw_ticks - one.cpu_raw_ticks, 0.003)
        self.assertAlmostEqual(four.reformation_raw_ticks - one.reformation_raw_ticks, 0.0015)
        self.assertEqual(one.details["workload"]["traffic_bytes"], four.details["workload"]["traffic_bytes"])

    def test_contiguous_expert_fastpath_skips_gpu_only_and_attention_still_packs(self):
        model = BatchOverheadModel(preset="structural")
        contiguous = replace(self.workload, fragments=1, contiguous=True)
        ordinary = model.evaluate("expert", 1, 128, replace(contiguous, contiguous=False))
        expert = model.evaluate("expert", 1, 128, contiguous)
        attention = model.evaluate("attention", 1, 128, contiguous)
        self.assertEqual(expert.cpu_raw_ticks, ordinary.cpu_raw_ticks)
        self.assertGreater(expert.cpu_raw_ticks, 0)
        self.assertEqual(expert.gpu_ticks, 1)
        self.assertEqual(expert.reformation_raw_ticks, 0)
        self.assertTrue(expert.details["components"]["fast_path"])
        self.assertEqual(expert.details["components"]["executed_traffic_bytes"], 0)
        self.assertGreater(attention.reformation_raw_ticks, 0)
        self.assertFalse(attention.details["components"]["fast_path"])

    def test_units_scale_and_exposure_apply_once(self):
        for ticks_per_ms in (1, 10, 1000):
            with self.subTest(ticks_per_ms=ticks_per_ms):
                model = BatchOverheadModel(
                    preset="structural", ticks_per_millisecond=ticks_per_ms,
                    cpu_schedule_scale=2, cpu_schedule_overlap=.25, reformation_overlap=.5)
                estimate = model.evaluate("attention", ticks_per_ms, 32, self.workload)
                self.assertAlmostEqual(estimate.cpu_raw_ticks / ticks_per_ms, .02672)
                self.assertAlmostEqual(estimate.cpu_exposed_ticks / ticks_per_ms, .02004)
                self.assertAlmostEqual(estimate.reformation_raw_ticks / ticks_per_ms, .00451072)
                self.assertAlmostEqual(estimate.reformation_exposed_ticks / ticks_per_ms, .00225536)
                self.assertAlmostEqual(estimate.total_ticks / ticks_per_ms, 1.02229536)

    def test_zero_coefficients_and_fully_hidden_costs(self):
        zero = {name: 0 for name in STRUCTURAL_DEFAULTS
                if name not in ("reformation_bandwidth_gb_s", "reformation_scale")}
        model = BatchOverheadModel(preset="structural", **zero)
        estimate = model.evaluate("attention", 1, 32, self.workload)
        self.assertEqual(estimate.cpu_raw_ticks, 0)
        self.assertAlmostEqual(estimate.reformation_raw_ticks, .0131072)
        hidden = BatchOverheadModel(preset="structural", cpu_schedule_overlap=1, reformation_overlap=1)
        estimate = hidden.evaluate("attention", 1, 32, self.workload)
        self.assertEqual(estimate.total_ticks, 1)
        self.assertGreater(estimate.cpu_raw_ticks, 0)
        self.assertGreater(estimate.reformation_raw_ticks, 0)

    def test_reformation_scale_changes_raw_cost_before_overlap_without_touching_cpu(self):
        for ticks_per_ms in (1, 10, 1000):
            for scale in (0, 1, 2):
                with self.subTest(ticks_per_ms=ticks_per_ms, scale=scale):
                    model = BatchOverheadModel(
                        preset="structural", ticks_per_millisecond=ticks_per_ms,
                        reformation_scale=scale, reformation_overlap=.25)
                    estimate = model.evaluate("attention", ticks_per_ms, 32, self.workload)
                    self.assertAlmostEqual(estimate.cpu_raw_ticks / ticks_per_ms, .01336)
                    self.assertAlmostEqual(estimate.reformation_raw_ticks / ticks_per_ms, .00451072 * scale)
                    self.assertAlmostEqual(estimate.reformation_exposed_ticks / ticks_per_ms, .00451072 * scale * .75)
                    components = estimate.details["components"]
                    self.assertAlmostEqual(components["reformation_unscaled_us"], 4.51072)
                    self.assertEqual(components["reformation_scale"], scale)
                    self.assertAlmostEqual(components["reformation_raw_us"], 4.51072 * scale)
                    self.assertAlmostEqual(components["reformation_fixed_us"], 3)
                    self.assertAlmostEqual(components["reformation_bytes_us"], 1.31072)
                    self.assertAlmostEqual(components["reformation_fragments_us"], .2)
                    self.assertFalse(components["fast_path"])
                    self.assertEqual(components["executed_traffic_bytes"], 655360 if scale else 0)
                    self.assertEqual(estimate.details["workload"]["traffic_bytes"], 655360)
                    json.dumps(estimate.details, allow_nan=False)
        contiguous = replace(self.workload, fragments=1, contiguous=True)
        expert = BatchOverheadModel(preset="structural", reformation_scale=2).evaluate(
            "expert", 1, 128, contiguous)
        self.assertTrue(expert.details["components"]["fast_path"])
        self.assertEqual(expert.details["components"]["reformation_unscaled_us"], 0)
        self.assertEqual(expert.details["components"]["executed_traffic_bytes"], 0)
        self.assertEqual(expert.reformation_raw_ticks, 0)

    def test_reformation_scaled_estimates_cannot_overflow(self):
        for overlap in (0, 1):
            model = BatchOverheadModel(
                preset="structural", reformation_scale=1e308, reformation_overlap=overlap)
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                model.evaluate("attention", 1, 32, self.workload)

    def test_finalize_and_empty_batches_need_no_workload_and_charge_nothing(self):
        model = BatchOverheadModel(preset="structural")
        for stage, size in (("finalize", 4), ("attention", 0), ("expert", 0)):
            with self.subTest(stage=stage):
                estimate = model.evaluate(stage, 1, size)
                self.assertEqual(estimate.total_ticks, 1)
                self.assertIsNone(estimate.details)
        for stage in ("attention", "expert"):
            for workload in (None, {}, 1):
                with self.subTest(stage=stage, workload=workload), self.assertRaises(ValueError):
                    model.evaluate(stage, 1, 1, workload)

    def test_workload_counts_are_strict_and_frozen(self):
        for name in ("input_rows", "fragments", "candidate_queues", "hidden_dim", "bytes_per_element"):
            for value in (-1, 1.0, True, None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(self.workload, **{name: value})
        for name in ("hidden_dim", "bytes_per_element"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(self.workload, **{name: 0})
        with self.assertRaises(ValueError):
            replace(self.workload, contiguous=1)
        with self.assertRaises(FrozenInstanceError):
            self.workload.fragments = 8

    def test_impossible_nonempty_workloads_fail(self):
        invalid = (replace(self.workload, input_rows=31),
                   replace(self.workload, fragments=0),
                   replace(self.workload, fragments=129),
                   replace(self.workload, candidate_queues=0),
                   replace(self.workload, contiguous=True))
        model = BatchOverheadModel(preset="structural")
        for workload in invalid:
            with self.subTest(workload=workload), self.assertRaises(ValueError):
                model.evaluate("attention", 1, 32, workload)

    def test_coefficients_and_mode_overrides_cannot_be_silently_ignored(self):
        for name in STRUCTURAL_DEFAULTS:
            for value in (-1, math.nan, math.inf, True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    BatchOverheadModel(preset="structural", **{name: value})
            for preset in ("none", "paper"):
                with self.subTest(name=name, preset=preset), self.assertRaises(ValueError):
                    BatchOverheadModel(preset=preset, **{name: STRUCTURAL_DEFAULTS[name]})
        with self.assertRaises(ValueError):
            BatchOverheadModel(preset="structural", reformation_bandwidth_gb_s=0)
        with self.assertRaises(ValueError):
            BatchOverheadModel(preset="structural", reformation_compute_ratio=.05)
        BatchOverheadModel(preset="structural", reformation_compute_ratio=0)
        huge = BatchOverheadModel(preset="structural", cpu_fixed_us=1e308, cpu_schedule_scale=10)
        with self.assertRaises(ValueError):
            huge.evaluate("attention", 1, 32, self.workload)

    def test_config_roundtrip_and_old_preset_keys_remain_unchanged(self):
        original_keys = {"preset", "cpu_schedule_scale", "cpu_schedule_overlap",
                         "reformation_compute_ratio", "reformation_overlap"}
        for preset in ("none", "paper", "structural"):
            with self.subTest(preset=preset):
                model = BatchOverheadModel(preset=preset)
                config = model.config()
                expected = original_keys | set(STRUCTURAL_DEFAULTS) if preset == "structural" else original_keys
                self.assertEqual(set(config), expected)
                self.assertEqual(BatchOverheadModel(**config).config(), config)
                if preset != "structural":
                    self.assertIsNone(model.evaluate("attention", 1, 32).details)
        description = BatchOverheadModel(preset="structural").describe()
        self.assertFalse(description["hardware_calibrated"])
        self.assertEqual(description["cpu_hidden_fraction_applied"], {"attention": 0, "expert": 0})
        self.assertIn("assumptions", description["initialization_rationale"])
        self.assertIn("no independent CPU resource", description["overlap_semantics"])
        json.dumps(description, allow_nan=False)

    def test_cli_supports_each_coefficient_without_changing_old_defaults(self):
        parser = argparse.ArgumentParser()
        add_overhead_arguments(parser)
        self.assertEqual(overhead_config_from_args(parser.parse_args([])), BatchOverheadModel().config())
        argv = ["--overhead-model", "structural", "--cpu-schedule-overlap", ".4"]
        for name in STRUCTURAL_DEFAULTS:
            argv.extend(["--" + name.replace("_", "-"), "2"])
        config = overhead_config_from_args(parser.parse_args(argv))
        self.assertEqual(config["preset"], "structural")
        self.assertEqual(config["cpu_schedule_overlap"], .4)
        for name in STRUCTURAL_DEFAULTS:
            self.assertEqual(config[name], 2)
        with self.assertRaises(ValueError):
            overhead_config_from_args(parser.parse_args(["--cpu-fixed-us", "10"]))


if __name__ == "__main__":
    unittest.main()
