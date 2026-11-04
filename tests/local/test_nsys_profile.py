"""Configuration safety checks that do not import CUDA or Ray."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


_source = Path(__file__).resolve().parents[2] / "disagmoe/utils/nsys_profile.py"
_spec = importlib.util.spec_from_file_location("nsys_profile", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
make_nsight_config = _module.make_nsight_config
collect_nsight_reports = _module.collect_nsight_reports


class NsightConfigTests(unittest.TestCase):
    def test_finalizing_a_session_does_not_request_worker_termination(self):
        # These options are coupled: session shutdown must never signal the
        # worker, which still has Ray RPCs and cleanup to finish afterwards.
        for cpu_sampling in (False, True):
            with self.subTest(cpu_sampling=cpu_sampling):
                config = make_nsight_config(cpu_sampling)
                self.assertEqual(config["capture-range-end"], "stop-shutdown")
                self.assertEqual(config["kill"], "none")
                self.assertEqual(config["wait"], "primary")

    def test_ray_output_mutation_does_not_leak_into_later_workers(self):
        first = make_nsight_config()
        # Ray's plugin adds a session-specific output path to the supplied dict.
        first["o"] = "/tmp/old-ray-session/logs/nsight/worker_process_%p"
        self.assertNotIn("o", make_nsight_config())

    def test_cpu_sampling_requires_an_explicit_boolean(self):
        # A string such as "false" is truthy and must not silently enable perf.
        for invalid in ("false", "true", 0, 1, None):
            with self.subTest(value=invalid), self.assertRaises(TypeError):
                make_nsight_config(invalid)


class NsightReportTests(unittest.TestCase):
    def test_collects_unique_nonempty_reports_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ray"
            session = root / "session_test"
            reports = session / "logs/nsight"
            reports.mkdir(parents=True)
            (root / "session_latest").symlink_to(session, target_is_directory=True)
            first = reports / "worker_process_100.nsys-rep"
            second = reports / "worker_process_200.nsys-rep"
            first.write_bytes(b"first worker report")
            second.write_bytes(b"second worker report")
            (reports / "worker_process_300.nsys-rep").touch()
            output = Path(tmp) / "run/nsight"

            copied = collect_nsight_reports(root, output, expected_workers=2, timeout_s=4)

            self.assertEqual(len(copied), 2)
            self.assertTrue(all(Path(path).is_absolute() for path in copied))
            self.assertEqual({Path(path).read_bytes() for path in copied},
                             {first.read_bytes(), second.read_bytes()})
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            with self.assertRaises(FileExistsError):
                collect_nsight_reports(root, output, expected_workers=2, timeout_s=0)

    def test_timeout_reports_unique_count_and_source_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ray"
            session = root / "session_test"
            reports = session / "logs/nsight"
            reports.mkdir(parents=True)
            (root / "session_latest").symlink_to(session, target_is_directory=True)
            source = reports / "worker_process_100.nsys-rep"
            source.write_bytes(b"only one worker report")
            output = Path(tmp) / "run/nsight"

            with self.assertRaises(TimeoutError) as caught:
                collect_nsight_reports(root, output, expected_workers=4, timeout_s=0)

            self.assertIn("found 1 nonempty", str(caught.exception))
            self.assertIn(str(root), str(caught.exception))
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), b"only one worker report")


if __name__ == "__main__":
    unittest.main()
