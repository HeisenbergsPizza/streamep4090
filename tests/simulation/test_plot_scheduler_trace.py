"""Integrity and time-weighting checks for scheduler trace figures."""

import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from simulation.plot_scheduler_trace import (
    common_window, crop_trace, execution_intervals, load_trace, main, window_summary,
)
from simulation.queue_trace import QueueTraceRecorder


def fixture(directory, policy="defrag", edges=(0., 1., 3., 4.), expert_scheduling="layer"):
    suffix = "" if expert_scheduling == "layer" else f"_{expert_scheduling}"
    path = Path(directory) / f"{policy}{suffix}.npz"
    queue = np.zeros((3, 2, 5), dtype=np.float64)
    queue[:, 0, 0] = (2, 4, 0)
    queue[:, 0, 1] = (2, 0, 4)
    queue[:, :, 2] = 100  # Finalize must never increase active model layers.
    queue[:, 0, 3] = (0, 0, 2)
    queue[:, 0, 4] = (0, 8, 0)
    active = np.array([[1., 0], [2., 0], [2., 0]])
    attention = np.array([[1., 0], [1., 0], [1., 0]])
    expert = np.array([[0., 0], [1., 0], [1., 0]])
    # In bin 0 the two attention queues alternate, so the mean count is one,
    # although both queue-depth means are positive. This distinction matters.
    np.savez(path, time_edges_ms=np.array(edges), queue_mean=queue,
             active_layers_mean=active, attention_active_layers_mean=attention,
             expert_active_layers_mean=expert)
    decisions = path.with_suffix(".decisions.jsonl.gz")
    records = [dict(gpu=0, stage="attention", layer=0, start_ms=.2, end_ms=1.2),
               dict(gpu=0, stage="expert", layer=1, start_ms=1.2, end_ms=2.5),
               dict(gpu=0, stage="finalize", layer=2, start_ms=2.5, end_ms=2.9),
               dict(gpu=0, stage="expert", layer=0, start_ms=2.9, end_ms=5.),
               dict(gpu=1, stage="attention", layer=0, start_ms=.1, end_ms=.3)]
    with gzip.open(decisions, "wt") as file:
        for record in records:
            if expert_scheduling == "expert" and record["stage"] == "expert":
                record["expert_id"] = 3
            file.write(json.dumps(record) + "\n")
    metadata = {"format": "streaminfer.queue_trace", "version": 1, "policy": policy,
                "complete": True, "status": "complete",
                "num_layers": 2, "num_gpus": 2, "top_k": 2, "shape": [3, 2, 5],
                "queue_labels": ["A_0", "A_1", "A_2", "E_0", "E_1"],
                "finalize_queue_index": 2, "ticks_per_millisecond": 10,
                "paths": {"decisions": str(decisions)}}
    if expert_scheduling != "layer":
        metadata["simulation_config"] = {"expert_scheduling": expert_scheduling}
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    return path


class PlotSchedulerTraceTests(unittest.TestCase):
    def test_duration_weighted_summary_preserves_gpu_and_excludes_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = load_trace(fixture(directory))
            rows = window_summary(trace)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["duration_ms"], 4)
            self.assertAlmostEqual(rows[0]["active_ready_layers_time_mean"], 1.75)
            self.assertAlmostEqual(rows[0]["attention_active_layers_time_mean"], 1.)
            self.assertAlmostEqual(rows[0]["expert_active_layers_time_mean"], .75)
            self.assertAlmostEqual(rows[0]["attention_ready_tokens_time_mean"], 4.)
            self.assertAlmostEqual(rows[0]["expert_ready_copies_time_mean"], 4.5)
            self.assertAlmostEqual(rows[0]["expert_ready_token_equivalents_time_mean"], 2.25)
            self.assertEqual(rows[1]["active_ready_layers_time_mean"], 0)
            self.assertEqual(rows[1]["finalize_ready_tokens_time_mean"], 100)

    def test_complete_bin_crop_and_actual_execution_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = crop_trace(load_trace(fixture(directory)), 1., 3.)
            self.assertEqual(trace.queue_mean.shape, (1, 2, 5))
            row = window_summary(trace)[0]
            self.assertEqual(row["active_ready_layers_time_mean"], 2)
            self.assertEqual(row["expert_ready_token_equivalents_time_mean"], 4)
            intervals = execution_intervals(trace, 0)
            self.assertEqual(intervals["attention"], [(1., 1.2, 0)])
            self.assertEqual(intervals["expert"], [(1.2, 2.5, 1), (2.9, 3., 0)])
            self.assertNotIn("finalize", intervals)
            with self.assertRaisesRegex(ValueError, "partial bins"):
                crop_trace(trace, 1.1, 3.)
            with self.assertRaisesRegex(ValueError, "end_ms must exceed"):
                crop_trace(trace, 3., 1.)

    def test_gpu_execution_excludes_cpu_gap_and_retains_reformation(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "overhead"
            recorder = QueueTraceRecorder(
                prefix, num_gpus=1, num_layers=1, ticks_per_millisecond=4,
                interval_ms=1, policy="defrag", top_k=1,
                configuration={"batch_overhead_model": {"enabled": True}})
            recorder.capture_schedule(dict(
                gpu=0, stage="attention", layer=0, start=0, end=12,
                gpu_start=6, compute_start=8, compute_end=12))
            recorder.finish(12)
            trace = load_trace(prefix.with_suffix(".npz"))
            # CPU gap [0,1.5), reformation [1.5,2), base compute [2,3).
            self.assertEqual(execution_intervals(trace, 0)["attention"], [(1.5, 3., 0)])
            self.assertEqual(execution_intervals(crop_trace(trace, 0., 1.), 0)["attention"], [])
            self.assertEqual(execution_intervals(crop_trace(trace, 1., 2.), 0)["attention"],
                             [(1.5, 2., 0)])
            self.assertIn("exposed CPU gap", trace.metadata["decision_window"])
            self.assertIn("when reserved", trace.metadata["queue_definition"])

    def test_gpu_start_uses_metadata_units_and_rejects_invalid_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = load_trace(fixture(directory))
            path = trace.path.with_suffix(".decisions.jsonl.gz")
            record = dict(gpu=0, stage="attention", layer=0, start_ms=.2, end_ms=1.2,
                          gpu_start=7)
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps(record) + "\n")
            self.assertEqual(execution_intervals(trace, 0)["attention"], [(.7, 1.2, 0)])
            for update in ({"gpu_start": 1}, {"gpu_start": 13}, {"gpu_start": True},
                           {"gpu_start": float("nan")}, {"ticks_per_millisecond": 0}):
                with self.subTest(update=update):
                    with gzip.open(path, "wt") as handle:
                        handle.write(json.dumps({**record, **update}) + "\n")
                    with self.assertRaises(ValueError):
                        execution_intervals(trace, 0)

    def test_default_comparison_uses_common_complete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            one = load_trace(fixture(directory, "defrag"))
            two = load_trace(fixture(directory, "fifo", edges=(0., 1., 3., 3.5)))
            traces, start, end = common_window([one, two])
            self.assertEqual((start, end), (0., 3.))
            self.assertEqual([len(trace.time_edges_ms) for trace in traces], [3, 3])
            with self.assertRaisesRegex(ValueError, "partial bins"):
                common_window([one, two], 0, 3.5)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                common_window([one, one])

    def test_schema_shape_and_nonfinite_values_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(directory)
            metadata_path = path.with_suffix(".metadata.json")
            original = json.loads(metadata_path.read_text())
            for update in ({"version": 9}, {"complete": False, "status": "recording"},
                           {"shape": [3, 2, 6]}, {"top_k": 0},
                           {"queue_labels": ["E_0"]}, {"policy": ""},
                           {"simulation_config": {"expert_scheduling": "unknown"}}):
                with self.subTest(update=update):
                    metadata_path.write_text(json.dumps({**original, **update}))
                    with self.assertRaises(ValueError):
                        load_trace(path)
            metadata_path.write_text(json.dumps(original))
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["queue_mean"][0, 0, 0] = np.nan
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_trace(path)

    def test_active_union_validation_and_empty_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(directory)
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["active_layers_mean"][0, 0] = 0
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "union bounds"):
                load_trace(path)
            arrays["time_edges_ms"] = np.array([0.])
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                load_trace(path)

    def test_cli_plots_csvs_metadata_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = fixture(root), fixture(root, "fifo")
            out = root / "figures"
            args = ["--traces", str(second), str(first), "--output-dir", str(out),
                    "--gpu", "0", "--start-ms", "1", "--end-ms", "3", "--dpi", "60"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            for stem in ("scheduler_heatmap", "scheduler_execution", "active_layers"):
                for extension in ("png", "pdf"):
                    self.assertGreater((out / f"{stem}.{extension}").stat().st_size, 1000)
            manifest = json.loads((out / "plot_metadata.json").read_text())
            self.assertIn("Simulation proxy", manifest["title"])
            self.assertEqual(manifest["expert_scale"], "normalized")
            self.assertEqual(manifest["actual_window_ms"], {"start": 1., "end": 3.})
            self.assertEqual(manifest["color_maxima_before_log1p"], {"attention": 4., "expert": 4.})
            self.assertEqual([item["policy"] for item in manifest["traces"]], ["defrag", "fifo"])
            with (out / "window_summary.csv").open() as file:
                summary = list(csv.DictReader(file))
            self.assertEqual(len(summary), 4)  # Both GPUs, even though only GPU0 is plotted.
            with (out / "layer_ready_summary.csv").open() as file:
                layers = list(csv.DictReader(file))
            self.assertEqual(len(layers), 16)
            self.assertTrue(all(row["layer"] in ("0", "1") for row in layers))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertTrue((out / "scheduler_heatmap.pdf").is_file())

    def test_expert_variant_plots_separately_and_retains_explicit_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layer = fixture(root)
            expert = fixture(root, expert_scheduling="expert")
            traces, _, _ = common_window([load_trace(layer), load_trace(expert)])
            self.assertEqual([trace.policy for trace in traces], ["defrag", "defrag/expert"])
            self.assertEqual(execution_intervals(traces[1], 0)["expert"],
                             [(1.2, 2.5, 1), (2.9, 4., 0)])
            for labels in (None, ["baseline", "individual"]):
                output = root / ("default" if labels is None else "custom")
                args = ["--traces", str(layer), str(expert), "--output-dir", str(output), "--dpi", "40"]
                if labels:
                    args.extend(["--labels", *labels])
                with contextlib.redirect_stdout(io.StringIO()):
                    main(args)
                manifest = json.loads((output / "plot_metadata.json").read_text())
                self.assertEqual([item["policy"] for item in manifest["traces"]],
                                 labels or ["defrag", "defrag/expert"])
                self.assertEqual([item["base_policy"] for item in manifest["traces"]], ["defrag", "defrag"])
                self.assertEqual([item["expert_scheduling"] for item in manifest["traces"]], ["layer", "expert"])
                self.assertEqual(manifest["execution_coordinates"], "layer")
                self.assertIn("expert_id", manifest["expert_identity_source"])
                with (output / "window_summary.csv").open() as handle:
                    self.assertEqual({row["expert_scheduling"] for row in csv.DictReader(handle)}, {"layer", "expert"})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--traces", str(layer), str(expert), "--labels", "one",
                      "--output-dir", str(root / "invalid")])
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
