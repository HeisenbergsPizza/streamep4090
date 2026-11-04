"""Exercise the real capture lifecycle without importing Ray, torch, or CUDA.

Only the methods under test are compiled from their source AST. Dependency
stand-ins expose capture ordering and error handling on CPU-only machines.
"""
import ast
import asyncio
from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]


def _compile_nodes(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<nsys-lifecycle-source>", "exec"), namespace)
    return namespace


class EngineCaptureTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.active_device = None
        self.start_error = None
        self.stop_error = None

        @contextmanager
        def device_context(device):
            self.assertIsNone(self.active_device)
            self.active_device = device
            self.calls.append(("enter", device))
            try:
                yield
            finally:
                self.calls.append(("exit", device))
                self.active_device = None

        def start():
            self.assertEqual(self.active_device, "cuda:0")
            self.calls.append("start")
            if self.start_error is not None:
                raise self.start_error

        def stop():
            self.assertEqual(self.active_device, "cuda:0")
            self.calls.append("stop")
            if self.stop_error is not None:
                raise self.stop_error

        tree = ast.parse((REPO / "disagmoe/frontend/engine.py").read_text())
        engine = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "Engine")
        methods = [node for node in engine.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name in ("start_nsys_profile", "stop_nsys_profile")]
        probe = ast.ClassDef(name="EngineProbe", bases=[], keywords=[],
                             body=methods, decorator_list=[])
        namespace = _compile_nodes([probe], {
            "os": os,
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                device=device_context,
                profiler=SimpleNamespace(start=start, stop=stop))),
        })
        self.engine = namespace["EngineProbe"]()
        self.engine.device = "cuda:0"
        self.engine.device_id = 2
        self.engine._nsys_profile_active = False

    def test_idempotent_start_stop_and_worker_mapping(self):
        self.engine.stop_nsys_profile()
        self.assertEqual(self.calls, [])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
            first = self.engine.start_nsys_profile()
            second = self.engine.start_nsys_profile()
        self.assertEqual(first, second)
        self.assertEqual(first, {
            "pid": os.getpid(), "device_id": 2,
            "cuda_visible_devices": "2", "cuda_device": "cuda:0",
        })
        self.assertTrue(self.engine._nsys_profile_active)
        self.engine.stop_nsys_profile()
        self.engine.stop_nsys_profile()
        self.assertFalse(self.engine._nsys_profile_active)
        self.assertEqual(self.calls, [
            ("enter", "cuda:0"), "start", ("exit", "cuda:0"),
            ("enter", "cuda:0"), "stop", ("exit", "cuda:0"),
        ])

    def test_start_failure_leaves_inactive_and_allows_retry(self):
        self.start_error = RuntimeError("capture start failed")
        with self.assertRaises(RuntimeError) as caught:
            self.engine.start_nsys_profile()
        self.assertIs(caught.exception, self.start_error)
        self.assertFalse(self.engine._nsys_profile_active)
        self.assertIsNone(self.active_device)
        self.start_error = None
        self.engine.start_nsys_profile()
        self.assertTrue(self.engine._nsys_profile_active)

    def test_stop_failure_preserves_active_and_allows_retry(self):
        self.engine.start_nsys_profile()
        self.stop_error = RuntimeError("capture stop failed")
        with self.assertRaises(RuntimeError) as caught:
            self.engine.stop_nsys_profile()
        self.assertIs(caught.exception, self.stop_error)
        self.assertTrue(self.engine._nsys_profile_active)
        self.assertIsNone(self.active_device)
        self.stop_error = None
        self.engine.stop_nsys_profile()
        self.assertFalse(self.engine._nsys_profile_active)


class BenchmarkCaptureTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.run_error = None
        self.start_error = None
        self.stop_error = None
        self.metrics = object()

        def event(name, failure_attr=None):
            def action(*args, **kwargs):
                self.events.append(name)
                if failure_attr and getattr(self, failure_attr) is not None:
                    raise getattr(self, failure_attr)
            return action

        def async_event(name):
            async def action(*args, **kwargs):
                self.events.append(name)
            return action

        async def run_benchmark(*args, **kwargs):
            self.events.append("run")
            if self.run_error is not None:
                raise self.run_error
            return [], [], [], 1.0

        def post_benchmark(*args, **kwargs):
            self.events.append("post")
            return self.metrics

        self.master = SimpleNamespace(
            start_polling_results=event("poll"),
            start_scheduler=async_event("scheduler_start"),
            stop_scheduler=async_event("scheduler_stop"),
            start_nsys_profile=event("capture_start", "start_error"),
            stop_nsys_profile=event("capture_stop", "stop_error"),
            stop_workers=event("workers_stop"),
            stop_profile=event("torch_profile_stop"),
            reset=event("reset"),
        )
        tree = ast.parse((REPO / "benchmark/benchmark_serving.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.AsyncFunctionDef)
                        and node.name == "benchmark_serving")
        noop = lambda *args, **kwargs: None
        namespace = _compile_nodes([function], {
            "Controller": object,
            "benchmark_warmup": async_event("warmup"),
            "run_benchmark": run_benchmark,
            "post_benchmark": post_benchmark,
            "logger": SimpleNamespace(info=noop, warning=noop, error=noop, exception=noop),
        })
        self.benchmark = namespace["benchmark_serving"]
        self.args = SimpleNamespace(
            nsys=True, profile_dir=None, generator_type="uniform", num_requests=8,
            min_input_len=64, max_input_len=65,
            min_output_len=8, max_output_len=9, rate=1,
        )

    def run_benchmark(self):
        return asyncio.run(self.benchmark(self.master, self.args))

    def test_capture_only_after_warmup_and_before_worker_shutdown(self):
        self.assertIs(self.run_benchmark(), self.metrics)
        self.assertEqual(self.events, [
            "poll", "scheduler_start", "warmup", "capture_start", "run",
            "capture_stop", "scheduler_stop", "workers_stop", "torch_profile_stop",
            "post", "reset",
        ])

    def test_disabled_does_not_call_cuda_profiler(self):
        self.args.nsys = False
        self.assertIs(self.run_benchmark(), self.metrics)
        self.assertNotIn("capture_start", self.events)
        self.assertNotIn("capture_stop", self.events)
        self.assertLess(self.events.index("warmup"), self.events.index("run"))

    def test_run_failure_and_cancellation_survive_cleanup_failure(self):
        for error in (RuntimeError("inference failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                self.events.clear()
                self.run_error = error
                self.stop_error = RuntimeError("cleanup failed")
                async def capture_failure():
                    # Catch inside the task: asyncio.run may reconstruct a
                    # CancelledError when returning a cancelled task's result.
                    try:
                        await self.benchmark(self.master, self.args)
                    except BaseException as caught:
                        return caught
                    self.fail("benchmark failure was swallowed")
                self.assertIs(asyncio.run(capture_failure()), error)
                self.assertEqual(self.events[-2:], ["run", "capture_stop"])
                self.assertNotIn("post", self.events)

    def test_capture_start_failure_survives_cleanup_failure(self):
        self.start_error = RuntimeError("capture start failed")
        self.stop_error = RuntimeError("cleanup failed")
        with self.assertRaises(RuntimeError) as caught:
            self.run_benchmark()
        self.assertIs(caught.exception, self.start_error)
        self.assertEqual(self.events[-2:], ["capture_start", "capture_stop"])
        self.assertNotIn("run", self.events)

    def test_stop_failure_after_success_is_not_silenced(self):
        self.stop_error = RuntimeError("capture stop failed")
        with self.assertRaises(RuntimeError) as caught:
            self.run_benchmark()
        self.assertIs(caught.exception, self.stop_error)
        self.assertNotIn("post", self.events)


if __name__ == "__main__":
    unittest.main()
