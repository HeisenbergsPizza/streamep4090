"""Finite-window overload rules and artifact-level capacity scan checks."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from simulation.steady_analysis import (
    Thresholds, analyze, assessment_blocks, capacity_brackets, classify_blocks, load_case, main,
    select_representative_cases,
)


def blocks(backlog, waiting, step_ms=5000):
    return [{"start_ms": index * step_ms, "end_ms": (index + 1) * step_ms,
             "active_requests": b - w, "waiting_requests": w, "outstanding_requests": b}
            for index, (b, w) in enumerate(zip(backlog, waiting))]


def case_fixture(root, policy="defrag", rate=10, seed=1, overloaded=False,
                 horizon_ms=40000, warmup_ms=10000, expert_scheduling="layer"):
    suffix = "" if expert_scheduling == "layer" else f"_{expert_scheduling}"
    directory = Path(root) / f"{policy}_{rate}_{seed}_{horizon_ms}{suffix}"
    directory.mkdir()
    timeline = []
    for index in range(horizon_ms // 5000):
        first, last = index * 5000, (index + 1) * 5000
        arrivals = rate * 5
        growth = 10 if overloaded and first >= warmup_ms else 0
        prior_wait = max(0, (first - warmup_ms) // 5000) * 10 if overloaded else 0
        start_state = dict(active_requests=128, waiting_requests=prior_wait, outstanding_requests=128 + prior_wait)
        end_state = dict(active_requests=128, waiting_requests=prior_wait + growth, outstanding_requests=128 + prior_wait + growth)
        means = {key: (start_state[key] + end_state[key]) / 2 for key in start_state}
        timeline.append({"start_ms": first, "end_ms": last, "observed_duration_ms": 5000,
                         "coverage_complete": True, "state_mean": means,
                         "state_at_start": start_state, "state_at_end": end_state,
                         "counts": {"arrived_requests": arrivals, "admitted_requests": arrivals - growth,
                                    "completed_requests": arrivals - growth, "completed_tokens": 32 * (arrivals - growth)}})
    selected = [item for item in timeline if item["start_ms"] >= warmup_ms]
    counts = {key: sum(item["counts"][key] for item in selected) for key in selected[0]["counts"]}
    window = {"start_ms": warmup_ms, "end_ms": horizon_ms,
              "observed_duration_ms": horizon_ms - warmup_ms, "coverage_complete": True,
              "state_at_start": selected[0]["state_at_start"], "state_at_end": selected[-1]["state_at_end"],
              "state_mean": {key: sum(item["state_mean"][key] for item in selected) / len(selected)
                             for key in selected[0]["state_mean"]}, "counts": counts,
              "gpu_busy_fraction": [.95, .97],
              "batch_mean": {"attention": 16, "expert_group": 80, "expert_individual": 4},
              "metrics_ms": {"itl": {"mean": 50., "p95": 60.}, "expert_ready_wait": {"mean": .3}}}
    configuration = {"policy": policy, "rate_rps": rate, "seed": seed,
                     "arrival_end_ms": horizon_ms, "warmup_ms": warmup_ms,
                     "comparison_configuration": {"gpus": 4, "output_tokens": 32, "active_request_cap": 128}}
    if expert_scheduling != "layer":
        configuration["expert_scheduling"] = expert_scheduling
    document = {"configuration": configuration,
                "simulation": {"steady_metrics": {"schema_version": 1, "time_unit": "ms",
                    "fully_drained": True, "windows": {"measurement": window}, "timeline": timeline}}}
    path = directory / "result.json"
    path.write_text(json.dumps(document))
    return path


class SteadyAnalysisTests(unittest.TestCase):
    def test_stable_overload_and_internal_cap_masking(self):
        stable = classify_blocks(blocks([128] * 6, [0] * 6), 50)
        self.assertEqual(stable["classification"], "stable")
        self.assertEqual(stable["drift_tolerance_rps"], 1)
        # Active stays at 128 in both cases; only external waiting reveals overload.
        overloaded = classify_blocks(blocks([128, 138, 148, 158, 168, 178], [0, 10, 20, 30, 40, 50]), 50)
        self.assertEqual(overloaded["classification"], "overload")
        self.assertAlmostEqual(overloaded["backlog_slope_rps"], 2)

    def test_transient_and_threshold_gray_zone(self):
        # Growing active occupancy without a growing waiting queue is a transient.
        result = classify_blocks(blocks([10, 20, 30, 40, 50, 60], [0] * 6), 10)
        self.assertEqual(result["classification"], "inconclusive")
        # Early accumulation followed by draining is not persistent overload.
        result = classify_blocks(blocks([128, 148, 168, 188, 148, 128], [0, 20, 40, 60, 20, 0]), 10)
        self.assertEqual(result["classification"], "inconclusive")
        self.assertEqual(classify_blocks(blocks([128, 128], [0, 0]), 10)["reason"], "insufficient_complete_blocks")
        result = classify_blocks(blocks([128, 138, 148, 158, 168, 178], [0, 10, 20, 30, 40, 50]), 50,
                                 Thresholds(absolute_drift_rps=3))
        self.assertEqual(result["classification"], "stable")

    def test_duration_weighted_reaggregation_and_partial_boundary_rejection(self):
        series = []
        for start, end, waiting in [(0, 1000, 0), (1000, 5000, 10)]:
            series.append({"start_ms": start, "end_ms": end, "coverage_complete": True,
                           "state_mean": {"active_requests": 128, "waiting_requests": waiting,
                                          "outstanding_requests": 128 + waiting}})
        result = assessment_blocks(series, 0, 5000, 5000)
        self.assertEqual(result[0]["waiting_requests"], 8)
        self.assertEqual(result[0]["outstanding_requests"], 136)
        with self.assertRaisesRegex(ValueError, "boundaries"):
            assessment_blocks(series, 500, 4500, 4000)

    def test_case_window_throughput_conservation_and_screen_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            case = load_case(case_fixture(directory, overloaded=True))
            row = case["summary"]
            self.assertEqual(row["completed_requests_per_sec"], 8)
            self.assertEqual(row["net_backlog_growth_rps"], 2)
            self.assertEqual(row["classification"], "overload")
            self.assertTrue(row["confirmation_eligible"])
            screen = load_case(case_fixture(directory, seed=2, horizon_ms=20000, warmup_ms=5000))
            self.assertEqual(screen["summary"]["classification"], "stable")
            self.assertFalse(screen["summary"]["confirmation_eligible"])
            path = Path(row["result_path"])
            data = json.loads(path.read_text())
            data["simulation"]["steady_metrics"]["windows"]["measurement"]["counts"]["arrived_requests"] += 1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "conservation"):
                load_case(path)

    def test_confirmed_bracket_requires_three_long_consistent_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = [load_case(case_fixture(directory, rate=rate, seed=seed, overloaded=(rate == 12)))
                     for rate in (10, 12) for seed in (1, 2, 3)]
            result = capacity_brackets(cases)["defrag"]
            self.assertEqual(result["status"], "empirical_bracket")
            self.assertEqual(result["stable_lower_rate_rps"], 10)
            self.assertEqual(result["overload_upper_rate_rps"], 12)
            self.assertEqual(capacity_brackets(cases[:1])["defrag"]["status"], "unresolved")

    def test_completion_markers_gate_confirmation_but_preserve_diagnostics(self):
        scenarios = [
            (False, None, None, "steady_metrics_not_fully_drained"),
            (True, {"fully_drained": False}, None, "result_validation_not_fully_drained"),
            (True, {"fully_drained": True}, {"status": "failed", "complete": False}, "status_not_complete"),
            (True, {"fully_drained": True}, {"status": "complete", "complete": False}, "status_not_complete"),
            (True, {"fully_drained": True}, {"status": "complete", "complete": True}, None),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for seed, (drained, validation, status, reason) in enumerate(scenarios):
                with self.subTest(seed=seed):
                    path = case_fixture(directory, seed=seed)
                    document = json.loads(path.read_text())
                    document["simulation"]["steady_metrics"]["fully_drained"] = drained
                    if validation is not None:
                        document["validation"] = validation
                    path.write_text(json.dumps(document))
                    if status is not None:
                        path.with_name("status.json").write_text(json.dumps(status))
                    case = load_case(path)
                    row = case["summary"]
                    self.assertEqual(row["classification"], "stable")
                    self.assertTrue(row["confirmation_duration_eligible"])
                    self.assertEqual(row["confirmation_eligible"], reason is None)
                    self.assertEqual(row["completion_validation_passed"], reason is None)
                    if reason is not None:
                        self.assertIn(reason, row["confirmation_ineligibility_reasons"])
                        point = capacity_brackets([case])["defrag"]["rate_points"][0]
                        self.assertEqual(point["confirmation"], "unvalidated_completion")
            # Even with three long, identically stable seeds, one failed drain
            # blocks the entire rate point from becoming confirmation evidence.
            valid = [load_case(case_fixture(directory, seed=seed)) for seed in (20, 21)]
            invalid = load_case(Path(directory) / "defrag_10_0_40000" / "result.json")
            result = capacity_brackets(valid + [invalid])["defrag"]
            self.assertIsNone(result["stable_lower_rate_rps"])
            self.assertIn("run_completion_not_validated", result["reasons"])

    def test_representative_selection_prefers_complete_long_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            short = load_case(case_fixture(directory, horizon_ms=20000, warmup_ms=5000))
            long = load_case(case_fixture(directory))
            failed_path = case_fixture(directory, horizon_ms=60000)
            document = json.loads(failed_path.read_text())
            document["validation"] = {"fully_drained": False}
            failed_path.write_text(json.dumps(document))
            failed = load_case(failed_path)
            for cases in ([short, failed, long], [long, failed, short]):
                selected, discarded = select_representative_cases(cases)
                self.assertEqual([case["summary"]["result_path"] for case in selected],
                                 [long["summary"]["result_path"]])
                self.assertEqual(len(discarded), 2)
                self.assertEqual({item["result_path"] for item in discarded},
                                 {short["summary"]["result_path"], str(failed_path.resolve())})
                self.assertTrue(all(item["selected_result_path"] == long["summary"]["result_path"]
                                    for item in discarded))

    def test_csv_plots_and_brackets_use_same_deduplicated_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cases"
            source.mkdir()
            for seed in (1, 2, 3):
                # A transient screen and a later confirmation of the same seed
                # must not be averaged or counted as separate observations.
                case_fixture(source, seed=seed, overloaded=True, horizon_ms=20000, warmup_ms=5000)
                case_fixture(source, seed=seed)
                case_fixture(source, rate=12, seed=seed, overloaded=True)
            output = root / "analysis"
            with patch("simulation.steady_analysis._save_figures") as save_figures, contextlib.redirect_stdout(io.StringIO()):
                report = analyze(source, output)
            plotted = save_figures.call_args.args[0]
            with (output / "case_summary.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(report["input_case_count"], 9)
            self.assertEqual(report["case_count"], 6)
            self.assertEqual(report["discarded_case_count"], 3)
            self.assertEqual(len(report["discarded_cases"]), 3)
            self.assertEqual({row["result_path"] for row in rows},
                             {case["summary"]["result_path"] for case in plotted})
            self.assertTrue(all(float(row["arrival_end_ms"]) == 40000 for row in rows))
            bracket = report["policies"]["defrag"]
            self.assertEqual(bracket["status"], "empirical_bracket")
            self.assertEqual(bracket["stable_lower_rate_rps"], 10)
            self.assertEqual(bracket["overload_upper_rate_rps"], 12)
            self.assertEqual(bracket["rate_points"][0]["completed_rps_mean"], 10)
            unselected = [load_case(path) for path in source.rglob("result.json")]
            self.assertEqual(capacity_brackets(unselected), report["policies"])

    def test_mixed_seeds_and_nonmonotone_patterns_remain_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = [load_case(case_fixture(directory, rate=10, seed=seed, overloaded=(seed == 3)))
                     for seed in (1, 2, 3)]
            result = capacity_brackets(cases)["defrag"]
            self.assertEqual(result["rate_points"][0]["confirmation"], "mixed_seeds")
            self.assertEqual(result["status"], "unresolved")
            cases += [load_case(case_fixture(directory, rate=12, seed=1, overloaded=True)),
                      load_case(case_fixture(directory, rate=14, seed=1))]
            result = capacity_brackets(cases)["defrag"]
            self.assertTrue(result["nonmonotone_observed"])
            self.assertEqual(result["status"], "unresolved")

    def test_different_physical_configurations_do_not_confirm(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [case_fixture(directory, seed=seed) for seed in (1, 2, 3)]
            data = json.loads(paths[2].read_text())
            data["configuration"]["comparison_configuration"]["active_request_cap"] = 64
            paths[2].write_text(json.dumps(data))
            result = capacity_brackets([load_case(path) for path in paths])["defrag"]
            self.assertEqual(result["rate_points"][0]["confirmation"], "mixed_configuration")

    def test_expert_and_layer_strategies_remain_separate_for_identical_policy_rate_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cases"
            source.mkdir()
            for seed in (1, 2, 3):
                case_fixture(source, seed=seed)
                case_fixture(source, seed=seed, expert_scheduling="expert", overloaded=True)
            with patch("simulation.steady_analysis._save_figures") as save_figures, contextlib.redirect_stdout(io.StringIO()):
                report = analyze(source, root / "analysis")
            self.assertEqual(report["case_count"], 6)
            self.assertEqual(report["discarded_case_count"], 0)
            self.assertEqual(set(report["policies"]), {"defrag", "defrag/expert"})
            self.assertEqual(report["policies"]["defrag"]["stable_lower_rate_rps"], 10)
            expert_point = report["policies"]["defrag/expert"]["rate_points"][0]
            self.assertEqual(expert_point["classification"], "overload")
            self.assertEqual(expert_point["confirmation"], "confirmed")
            rows = [case["summary"] for case in save_figures.call_args.args[0]]
            self.assertEqual({row["base_policy"] for row in rows}, {"defrag"})
            self.assertEqual({row["expert_scheduling"] for row in rows}, {"layer", "expert"})
            self.assertEqual(len({row["comparison_signature"] for row in rows}), 2)
            with (root / "analysis/case_summary.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 6)

    def test_inconsistent_or_unknown_expert_scheduling_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = case_fixture(directory, expert_scheduling="expert")
            original = json.loads(path.read_text())
            document = json.loads(path.read_text())
            document["configuration"]["expert_scheduling"] = "other"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "expert_scheduling"):
                load_case(path)
            original["configuration"]["comparison_configuration"]["expert_scheduling"] = "layer"
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_case(path)

    def test_cli_artifacts_stdout_and_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cases"
            source.mkdir()
            case_fixture(source, rate=10)
            case_fixture(source, rate=12, overloaded=True)
            output = root / "analysis"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(["--input-dir", str(source), "--output-dir", str(output)])
            self.assertIn("screen_only", stdout.getvalue())
            self.assertIn("not a mathematical", stdout.getvalue())
            report = json.loads((output / "capacity_brackets.json").read_text(),
                                parse_constant=lambda value: self.fail(f"invalid JSON constant {value}"))
            self.assertEqual(report["case_count"], 2)
            with (output / "case_summary.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            for name in ("throughput_vs_offered", "backlog_timeline", "scheduler_metrics"):
                for suffix in ("png", "pdf"):
                    self.assertGreater((output / f"{name}.{suffix}").stat().st_size, 1000)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--root", str(source), "--output-dir", str(output)])


if __name__ == "__main__":
    unittest.main()
