"""CPU regressions for streamed token counts and fixed-arrival window metrics."""
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/scripts/sglang"))
import bench_uniform as bench


async def lines(counts, done=True):
    for count in counts:
        event = {"meta_info": {"prompt_tokens": 64, "completion_tokens": count, "cached_tokens": 0}}
        yield b"data: " + json.dumps(event).encode() + b"\n"
        yield b"\n"
    if done:
        yield b"data: [DONE]\n"


class StreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_dummy_removes_one_token_from_coalesced_first_event(self):
        payloads = []

        class Request:
            async def __aenter__(self):
                return SimpleNamespace(raise_for_status=lambda: None, content=lines([2, 2, 4]))

            async def __aexit__(self, *args):
                pass

        def post(url, json):
            payloads.append(json)
            return Request()

        rows, elapsed = await bench.run_workload(
            SimpleNamespace(post=post), [[1] * 64, [2] * 64], 200, 3, first_bootstrap_room=11)
        self.assertEqual([p["bootstrap_room"] for p in payloads], [11, 12])
        for payload, row in zip(payloads, rows):
            self.assertEqual(payload["bootstrap_host"], "2.2.2.2")
            self.assertEqual(payload["sampling_params"]["max_new_tokens"], 4)
            self.assertTrue(payload["sampling_params"]["ignore_eos"])
            self.assertEqual(row["api_completion_tokens"], 4)
            self.assertEqual(len(row["token_times_s"]), 3)
            self.assertEqual(row["bootstrap_token_s"], row["token_times_s"][0])
            self.assertEqual(row["token_times_s"][1], row["token_times_s"][2])
        self.assertEqual(bench.summarize(rows, elapsed, (0, elapsed))["whole"]["output_tokens"], 6)

    async def test_open_loop_submits_before_first_response(self):
        all_sent = asyncio.Event()
        payloads = []

        class Request:
            async def __aenter__(self):
                await all_sent.wait()
                return SimpleNamespace(raise_for_status=lambda: None, content=lines([2]))

            async def __aexit__(self, *args):
                pass

        def post(url, json):
            payloads.append(json)
            if len(payloads) == 3:
                all_sent.set()
            return Request()

        observations, elapsed = await asyncio.wait_for(
            bench.run_workload(SimpleNamespace(post=post), [[i] * 64 for i in range(3)], 200, 2), 2)
        self.assertEqual([o["scheduled_s"] for o in observations], [0, .005, .01])
        self.assertTrue(all(len(o["token_times_s"]) == 2 for o in observations))
        self.assertLessEqual(max(o["response_s"] for o in observations), elapsed)
        self.assertTrue(all(p["sampling_params"]["ignore_eos"] for p in payloads))

    async def test_coalesced_and_repeated_chunks(self):
        t = iter(range(15))
        got = await bench.read_generation(lines([1, 3, 3]), 64, 3, lambda: next(t))
        self.assertEqual(got, ([0, 2, 2], 6, 0))

    async def test_incomplete_or_inconsistent_stream(self):
        for counts, done in [([1], True), ([1, 2], False), ([2, 1], True)]:
            with self.subTest(counts=counts, done=done), self.assertRaises((ValueError, RuntimeError)):
                await bench.read_generation(lines(counts, done), 64, 2, lambda: 0)


class MetricsTest(unittest.TestCase):
    def test_mode_cannot_mix_counting_conventions(self):
        normal = {"disaggregation_mode": "null", "load_format": "auto"}
        dummy = {"disaggregation_mode": "decode", "load_format": "dummy",
                 "disaggregation_transfer_backend": "fake"}
        bench.check_decode_mode(normal, False)
        bench.check_decode_mode(dummy, True)
        for info, mode in [(dummy, False), (normal, True), (dict(dummy, load_format="auto"), True),
                           (dict(dummy, disaggregation_transfer_backend="mooncake"), True),
                           (dict(dummy, speculative_algorithm="EAGLE"), True)]:
            with self.subTest(info=info, mode=mode), self.assertRaises(ValueError):
                bench.check_decode_mode(info, mode)

    def test_half_open_window_and_latency_cohorts(self):
        rows = [
            dict(scheduled_s=0, submitted_s=0, response_s=2, token_times_s=[1, 1.5]),
            dict(scheduled_s=1, submitted_s=1, response_s=3, token_times_s=[2, 2.75]),
        ]
        s = bench.summarize(rows, 3.001, [1, 2])
        self.assertEqual(s["window"]["submitted"], 1)
        self.assertEqual(s["window"]["completed"], 0)
        self.assertEqual(s["window"]["outstanding_start"], 1)
        self.assertEqual(s["window"]["outstanding_end"], 2)
        self.assertEqual(s["window"]["output_tokens"], 2)
        self.assertEqual(s["arrivals_in_window"]["requests"], 1)
        self.assertEqual(s["arrivals_in_window"]["tpot_ms"]["p50"], 750)
        self.assertEqual(s["whole"]["completed"], 2)
        self.assertEqual(s["whole"]["output_tokens"], 4)
        self.assertEqual(s["whole"]["outstanding_end"], 0)


if __name__ == "__main__":
    unittest.main()
