"""CPU checks for the actual benchmark lifecycle and its release-based clock."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[2]


def compile_function(name, namespace):
    tree = ast.parse((REPO / "benchmark/benchmark_serving.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<synchronized-benchmark>", "exec"), namespace)
    return namespace[name]


class SynchronizedBenchmarkLifecycleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        self.failures = {}
        self.responses = [object(), object()]
        self.metrics = object()

        def event(name, result=None):
            def action(*args, **kwargs):
                self.events.append(name)
                if name in self.failures:
                    raise self.failures[name]
                return result
            return action

        def async_event(name, result=None):
            async def action(*args, **kwargs):
                return event(name, result)(*args, **kwargs)
            return action

        self.master = SimpleNamespace(
            start_polling_results=event("poll"),
            start_scheduler=async_event("scheduler_start"),
            stop_scheduler=async_event("scheduler_stop"),
            prepare_synchronized_start=async_event("prepare_all", self.responses),
            abort_synchronized_start=async_event("abort"),
            start_nsys_profile=event("capture_start"),
            stop_nsys_profile=event("capture_stop"),
            stop_workers=event("workers_stop"),
            stop_profile=event("torch_profile_stop"),
            reset=event("reset"),
        )
        async def synchronized_run(master, args, responses):
            self.assertIs(responses, self.responses)
            return await async_event("release_and_run", ([], [], [], 1.0))()

        self.benchmark = compile_function("benchmark_serving", {
            "Controller": object,
            "benchmark_warmup": async_event("ordinary_warmup"),
            "run_benchmark": async_event("ordinary_run", ([], [], [], 1.0)),
            "run_synchronized_benchmark": synchronized_run,
            "post_benchmark": event("post", self.metrics),
            "logger": SimpleNamespace(info=lambda *a: None, exception=lambda *a: None),
        })
        self.args = SimpleNamespace(
            synchronized_start=True, nsys=True, profile_dir=None,
            num_nodes=1, num_gpus=4, dp_size=4, ep_size=4, tp_size=1,
            placement="colocate", max_batch_size_attn=16,
            generator_type="offline", num_requests=64,
            min_input_len=64, max_input_len=65, min_output_len=32, max_output_len=33,
            analyze_throughput=False, rate=1,
        )

    async def test_prepare_before_capture_release_after_capture(self):
        self.assertIs(await self.benchmark(self.master, self.args), self.metrics)
        self.assertEqual(self.events, [
            "poll", "scheduler_start", "ordinary_warmup", "prepare_all", "capture_start",
            "release_and_run", "capture_stop", "scheduler_stop", "workers_stop",
            "torch_profile_stop", "post", "reset",
        ])

    async def test_baseline_has_same_prepare_release_without_capture(self):
        self.args.nsys = False
        await self.benchmark(self.master, self.args)
        self.assertLess(self.events.index("prepare_all"), self.events.index("release_and_run"))
        self.assertNotIn("capture_start", self.events)
        self.assertNotIn("ordinary_run", self.events)

    async def test_failures_cancel_cohort_and_preserve_original_exception(self):
        for where in ("prepare_all", "capture_start", "release_and_run"):
            for error in (RuntimeError(where), asyncio.CancelledError()):
                with self.subTest(where=where, error=type(error).__name__):
                    self.events.clear()
                    self.failures = {where: error, "abort": RuntimeError("abort failure"),
                                     "capture_stop": RuntimeError("capture-stop failure")}
                    try:
                        await self.benchmark(self.master, self.args)
                    except BaseException as caught:
                        self.assertIs(caught, error)
                    else:
                        self.fail("original failure was swallowed")
                    self.assertEqual(self.events[-2:], ["abort", "capture_stop"])
                    self.assertNotIn("post", self.events)
                    if where != "release_and_run":
                        self.assertNotIn("release_and_run", self.events)

    async def test_api_server_and_invalid_cohort_fail_before_warmup(self):
        with self.assertRaises(ValueError):
            await self.benchmark(self.master, self.args, is_api_server=True)
        self.args.num_requests = 65
        with self.assertRaises(ValueError):
            await self.benchmark(self.master, self.args)
        self.assertEqual(self.events, [])


class SynchronizedBenchmarkClockTest(unittest.IsolatedAsyncioTestCase):
    async def test_elapsed_time_uses_release_target_and_finishes_before_reporting(self):
        events = []
        pbar = SimpleNamespace(close=lambda: events.append("close"))
        async def response(resp, finished, progress):
            self.assertIs(progress, pbar)
            self.assertIn("release", events)
            finished.append(resp)
            return resp

        async def release():
            events.append("release")
            return 40_000_000_000

        async def finish():
            events.append("finish")

        run = compile_function("run_synchronized_benchmark", {
            "Controller": object, "asyncio": asyncio,
            "time": SimpleNamespace(perf_counter=lambda: 41.5),
            "tqdm": SimpleNamespace(tqdm=lambda total: pbar), "process_response": response,
        })
        result = await run(SimpleNamespace(release_synchronized_start=release,
                                          finish_synchronized_start=finish), None, [41.25, 41.0])
        self.assertEqual(result, ([41.25, 41.0], [0.0, 0.0], [1.0, 1.25], 1.5))
        self.assertEqual(events, ["release", "finish", "close"])

    async def test_failed_release_cancels_response_waiters(self):
        events = []
        error = RuntimeError("release failed")
        async def response(*args):
            try:
                await asyncio.Event().wait()
            finally:
                events.append("waiter_cancelled")

        async def release():
            await asyncio.sleep(0)  # Let response tasks start waiting.
            raise error

        run = compile_function("run_synchronized_benchmark", {
            "Controller": object, "asyncio": asyncio,
            "tqdm": SimpleNamespace(tqdm=lambda total: SimpleNamespace(close=lambda: events.append("close"))),
            "process_response": response,
        })
        with self.assertRaises(RuntimeError) as caught:
            await run(SimpleNamespace(release_synchronized_start=release), None, [1, 2])
        self.assertIs(caught.exception, error)
        self.assertEqual(events, ["waiter_cancelled", "waiter_cancelled", "close"])


if __name__ == "__main__":
    unittest.main()
