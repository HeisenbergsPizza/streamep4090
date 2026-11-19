import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from experiments.scripts.rtx4090.plot_scheduler_choices import (
    load_run, main, outstanding_steps, read_choices, selected_window,
)


class TestSchedulerChoicePlot(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name)
        self.epoch = 10**16 + 1  # Epoch arithmetic must stay in integer nanoseconds.
        self.run = dict(format="streaminfer.scheduler_choices.v1", start_ns=self.epoch,
                        end_ns=self.epoch + 2_000_000_000, num_layers=4, device_ids=[0, 1],
                        rate=4, generator_type="uniform", submitted_s=[0, .25, .5],
                        finished_s=[.25, .9, 1.625])
        (self.path / "run.json").write_text(json.dumps(self.run))
        for gpu, rows in [(0, [(10_000_000, "attention", 2, 4), (10_000_000, "expert", 2, 7),
                               (10_900_000, "finalize", 4, 1), (11_000_000, "attention", 0, 2)]),
                          (1, [])]:
            with (self.path / f"choices-gpu-{gpu}.csv").open("w", newline="") as out:
                writer = csv.writer(out)
                writer.writerow(["time_ns", "stage", "layer", "batch_size"])
                for t, *meta in rows:
                    writer.writerow([self.epoch + t, *meta])

    def test_half_open_window_ties_and_finalize(self):
        run = load_run(self.path)
        window = selected_window(run, .01, 1)
        self.assertEqual(window, (self.epoch + 10_000_000, self.epoch + 11_000_000))
        self.assertEqual(read_choices(self.path, run, 0, *window), {
            "attention": [(0., 2)], "expert": [(0., 2)], "finalize": [(.9, 4)]})
        self.assertEqual(outstanding_steps(run)[1], [0, 1, 1, 2, 1, 0])

    def test_invalid_interval_and_corrupt_records(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            selected_window(self.run, 1, 1001)
        with (self.path / "choices-gpu-0.csv").open("a") as out:
            out.write(f"{self.epoch + 20_000_000},attention,4,1\n")
        with self.assertRaisesRegex(ValueError, "invalid stage"):
            read_choices(self.path, self.run, 0, *selected_window(self.run, .01, 1))

    def test_renderer_including_empty_rank_and_drain(self):
        try:
            import matplotlib.image as mpimg
        except ImportError:
            self.skipTest("matplotlib is required for the rendering test")
        for start, gpu_args, summary in [(.01, [], "GPU 0: 3, GPU 1: 0"),
                                         (1.0, ["--gpu", "1"], "includes drain")]:
            target = self.path / "out" / f"figure-{start}.png"
            log = io.StringIO()
            with redirect_stdout(log):
                main(["--run-dir", str(self.path), "--start-s", str(start), "--duration-ms", "1",
                      "--output", str(target), *gpu_args])
            self.assertIn(summary, log.getvalue())
            image = mpimg.imread(target)
            self.assertGreater(image.shape[0], 500)
            self.assertGreater(image[:, :, :3].std(), .02)


if __name__ == "__main__":
    unittest.main()
