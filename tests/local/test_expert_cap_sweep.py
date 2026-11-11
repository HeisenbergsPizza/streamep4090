"""CPU checks for sweep configuration, result aggregation, and failed trials."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "experiments/scripts/rtx4090/sweep_expert_cap.py"
spec = importlib.util.spec_from_file_location("expert_cap_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def metric_row(itl_ms=10.0):
    duration = itl_ms * 128 / 1000 + 0.1
    return {
        "num_requests": 512, "rate": 1, "DP_size": 4,
        "EP_size": 4, "num_experts": 128,
        "e2e_duration": duration,
        "req_throughput": 512 / duration,
        "token_throughput": 512 * 128 / duration,
        "req_latency_mean_ms": itl_ms * 127,
        "req_latency_median_ms": itl_ms * 127,
        "req_latency_p99_ms": itl_ms * 127 * 1.2,
        "itl_latency_mean_ms": itl_ms,
        "itl_latency_median_ms": itl_ms * 0.9,
        "itl_latency_p99_ms": itl_ms * 1.7,
        "peak_itl_median_ms": -1, "peak_itl_p99_ms": -1,
    }


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_row()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ExpertCapSweepTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="expert-cap-sweep-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output = self.directory / "results"

    def main_quietly(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return sweep.main(["--output-dir", str(self.output), *args])

    def test_actual_dry_run_validates_all_caps_without_output_or_profilers(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(self.directory))
        # Keep this CPU-only check independent of the caller's remote runtime paths.
        for name in ("DMOE_RAY_TMPDIR", "DMOE_RUNTIME_TMPDIR", "DMOE_TMPDIR"):
            env.pop(name, None)
        completed = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--dry-run", "--output-dir", str(self.output)],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        decoder = json.JSONDecoder()
        configs = []
        for match in re.finditer(r"(?m)^\{\s*\n\s*\"mode\"", completed.stdout):
            payload, _ = decoder.raw_decode(completed.stdout[match.start():])
            self.assertEqual(payload["mode"], "benchmark")
            configs.append(payload["configuration"])
        self.assertEqual([c["max_batch_size_expert"] for c in configs], [128, 256, 512, 1024])
        expected = {
            "num_requests": 512, "max_batch_size_attn": 128, "max_attn_graph_bsz": 128,
            "cuda_graph_attn": True, "cuda_graph_expert": True,
            "dp_size": 4, "ep_size": 4, "tp_size": 1,
            "num_layers": 48, "num_experts": 128, "topk": 8,
            "attn_qkv_quant": "none", "moe_linear_quant": "none",
            "synchronized_start": True, "generator_type": "offline",
            "unified_scheduler_type": "defrag", "gpu_usage": 0.80,
            "min_input_len": 64, "max_input_len": 65,
            "min_output_len": 128, "max_output_len": 129,
            "nsys": False, "profile_dir": None, "trace": False,
            "enable_advanced_logging": False,
        }
        for config in configs:
            with self.subTest(cap=config["max_batch_size_expert"]):
                for key, value in expected.items():
                    self.assertEqual(config[key], value, key)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.directory.iterdir()), [])
        self.assertEqual(completed.stdout.count("CLI validation passed"), 4)
        self.assertIn("PASS: all trial configurations validated", completed.stdout)

    def test_successful_trials_aggregate_medians_and_preserve_individual_results(self):
        measurements = {128: [10, 99, 30], 256: [8, 12, 10]}
        executed = []

        def run_one(trial, env):
            cap, repeat = trial["expert_cap"], trial["repeat"]
            executed.append((cap, repeat))
            self.assertEqual(env["PROFILE_RUN_DIR"], trial["run_dir"])
            self.assertEqual(env["PYTHON"], sys.executable)
            write_metrics(Path(trial["run_dir"]) / "metrics.csv",
                          [metric_row(measurements[cap][repeat - 1])])
            return 0

        with mock.patch.object(sweep, "run_one", side_effect=run_one):
            self.assertEqual(self.main_quietly("--caps", "128", "256", "--repeats", "3"), 0)

        self.assertEqual(set(executed), {(cap, repeat) for cap in measurements for repeat in (1, 2, 3)})
        self.assertEqual(len(executed), 6)
        # Trials must not always measure the same cap first.
        first_by_repeat = [next(cap for cap, rep in executed if rep == repeat) for repeat in (1, 2, 3)]
        self.assertGreater(len(set(first_by_repeat)), 1)
        runs = read_csv(self.output / "runs.csv")
        self.assertEqual(len(runs), 6)
        self.assertTrue(all(row["status"] == "ok" for row in runs))
        self.assertEqual(sorted(float(row["itl_latency_mean_ms"]) for row in runs), [8, 10, 10, 12, 30, 99])
        summaries = {int(row["expert_cap"]): row for row in read_csv(self.output / "summary.csv")}
        for cap, expected in ((128, 30), (256, 10)):
            row = summaries[cap]
            self.assertEqual(int(row["successful_runs"]), 3)
            self.assertEqual(int(row["failed_runs"]), 0)
            self.assertEqual(float(row["median_itl_latency_mean_ms"]), expected)
            self.assertEqual(float(row["median_req_latency_mean_ms"]), expected * 127)
            self.assertEqual(float(row["min_itl_latency_mean_ms"]), min(measurements[cap]))
            self.assertEqual(float(row["max_itl_latency_mean_ms"]), max(measurements[cap]))

    def test_failure_and_interruption_stop_remaining_caps_without_zero_latency(self):
        for failure in ("exit", "interrupt", "invalid_metrics", "truncated_metrics"):
            with self.subTest(failure=failure):
                self.output = self.directory / failure
                executed = []

                def run_one(trial, env):
                    cap = trial["expert_cap"]
                    executed.append(cap)
                    if cap == 128:
                        write_metrics(Path(trial["run_dir"]) / "metrics.csv", [metric_row()])
                        return 0
                    if failure == "interrupt":
                        raise KeyboardInterrupt
                    if failure == "invalid_metrics":
                        row = metric_row()
                        row["itl_latency_mean_ms"] = "nan"
                        write_metrics(Path(trial["run_dir"]) / "metrics.csv", [row])
                        return 0
                    if failure == "truncated_metrics":
                        path = Path(trial["run_dir"]) / "metrics.csv"
                        write_metrics(path, [metric_row()])
                        header, data = path.read_text().splitlines()
                        # The header is complete, but a writer stopped before the last metrics.
                        path.write_text(header + "\n" + ",".join(data.split(",")[:8]) + "\n")
                        return 0
                    return 42

                with mock.patch.object(sweep, "run_one", side_effect=run_one):
                    code = self.main_quietly()
                self.assertEqual(code, 130 if failure == "interrupt" else 1)
                self.assertEqual(executed, [128, 256])
                runs = read_csv(self.output / "runs.csv")
                self.assertEqual([row["expert_cap"] for row in runs], ["128", "256"])
                self.assertEqual(runs[0]["status"], "ok")
                self.assertEqual(runs[1]["status"], "interrupted" if failure == "interrupt" else "failed")
                self.assertTrue(runs[1]["error"])
                self.assertEqual(runs[1]["itl_latency_mean_ms"], "")
                summaries = read_csv(self.output / "summary.csv")
                self.assertEqual(summaries[1]["successful_runs"], "0")
                self.assertEqual(summaries[1]["failed_runs"], "1")
                self.assertEqual(summaries[1]["median_itl_latency_mean_ms"], "")

    def test_existing_output_is_preserved_and_no_trial_is_launched(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("previous experiment", encoding="utf-8")
        for args in ((), ("--dry-run",)):
            with self.subTest(args=args), mock.patch.object(sweep, "run_one") as run_one:
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    self.main_quietly(*args)
                run_one.assert_not_called()
        self.assertEqual(sentinel.read_text(), "previous experiment")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_rejects_wrong_workload_invalid_values_and_multiple_metric_rows(self):
        path = self.directory / "metrics.csv"
        cases = [
            ("wrong cohort", {"num_requests": 256}),
            ("wrong topology", {"EP_size": 2}),
            ("nonfinite latency", {"itl_latency_mean_ms": "nan"}),
            ("negative latency", {"req_latency_mean_ms": -1}),
            ("nonfinite throughput", {"token_throughput": "inf"}),
            ("zero duration", {"e2e_duration": 0}),
            ("missing latency", {"itl_latency_p99_ms": ""}),
        ]
        for name, overrides in cases:
            with self.subTest(name=name):
                write_metrics(path, [{**metric_row(), **overrides}])
                with self.assertRaises(ValueError):
                    sweep.read_metrics(path)
        for rows in ([], [metric_row(), metric_row()]):
            with self.subTest(rows=len(rows)):
                write_metrics(path, rows)
                with self.assertRaisesRegex(ValueError, "Expected one metrics row"):
                    sweep.read_metrics(path)

    def test_rejects_truncated_duplicate_overlong_and_malformed_csv(self):
        path = self.directory / "metrics.csv"
        fields = list(metric_row())
        values = [str(metric_row()[key]) for key in fields]
        header = ",".join(fields)
        data = ",".join(values)
        cases = {
            "truncated": header + "\n" + ",".join(values[:8]) + "\n",
            "duplicate": header + ",itl_latency_mean_ms\n" + data + ",10\n",
            "extra data": header + "\n" + data + ",unexpected\n",
            "malformed quote": header + '\n"unterminated\n',
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                path.write_text(text, encoding="utf-8")
                if name == "truncated":
                    self.assertIsNone(read_csv(path)[0]["itl_latency_mean_ms"])
                with self.assertRaises(ValueError):
                    sweep.read_metrics(path)

    @unittest.skipUnless(os.name == "posix", "Process sessions require POSIX")
    def test_cleanup_waits_for_separate_group_after_launcher_exits(self):
        # Fake IDs are never passed to the OS: all discovery, signal and sleep calls
        # are mocked. This models timeout creating another group in the same session.
        process = mock.Mock(pid=4200)
        process.poll.return_value = 0  # Launcher already exited, child group survives.
        process.wait.return_value = 0
        with mock.patch.object(sweep, "session_groups", side_effect=[
            {4200, 4300}, {4300}, set(), set(),
        ]) as groups, mock.patch.object(sweep.os, "killpg") as killpg, \
                mock.patch.object(sweep.time, "monotonic", side_effect=[0, 1, 2]), \
                mock.patch.object(sweep.time, "sleep") as sleep:
            sweep.stop_trial(process, grace_seconds=5)
        self.assertCountEqual(killpg.call_args_list, [
            mock.call(4200, signal.SIGTERM), mock.call(4300, signal.SIGTERM),
        ])
        self.assertGreaterEqual(process.poll.call_count, 2)
        self.assertTrue(all(call == mock.call(4200) for call in groups.call_args_list))
        sleep.assert_called_once()
        process.wait.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "Process sessions require POSIX")
    def test_cleanup_escalates_only_remaining_trial_groups(self):
        process = mock.Mock(pid=4200)
        process.poll.return_value = 0
        with mock.patch.object(sweep, "session_groups", side_effect=[
            {4300}, {4300},
        ]), mock.patch.object(sweep.os, "killpg") as killpg, \
                mock.patch.object(sweep.time, "monotonic", side_effect=[0, 6]):
            sweep.stop_trial(process, grace_seconds=5)
        self.assertEqual(killpg.call_args_list, [
            mock.call(4300, signal.SIGTERM), mock.call(4300, signal.SIGKILL),
        ])
        process.wait.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "Process sessions require POSIX")
    def test_group_discovery_excludes_other_sessions_zombies_and_exited_processes(self):
        listing = "4200 S\n4300 S\n4301 Z\n5000 S\n6000 S\n"

        def getsid(pid):
            if pid == 6000:
                raise ProcessLookupError
            return 4200 if pid in (4200, 4300) else 5000

        with mock.patch.object(sweep.subprocess, "run", return_value=mock.Mock(stdout=listing)), \
                mock.patch.object(sweep.os, "getsid", side_effect=getsid) as sid, \
                mock.patch.object(sweep.os, "getpgid", side_effect=lambda pid: pid) as pgid:
            self.assertEqual(sweep.session_groups(4200), {4200, 4300})
        self.assertNotIn(mock.call(4301), sid.call_args_list)
        self.assertEqual(pgid.call_args_list, [mock.call(4200), mock.call(4300)])


if __name__ == "__main__":
    unittest.main()
