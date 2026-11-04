import asyncio
import unittest

from disagmoe.frontend.async_result import AsyncResult
from disagmoe.frontend.request_stats import SloStat
from disagmoe.utils.runtime_checks import kv_cache_block_budget, validate_request_lengths


class AsyncResultTest(unittest.IsolatedAsyncioTestCase):
    async def test_completion_before_wait_is_not_lost(self):
        result = AsyncResult(1)
        stat = object()
        await result.put(stat)
        self.assertIs(await asyncio.wait_for(result.get(), 0.1), stat)

    async def test_all_waiters_receive_result(self):
        result = AsyncResult(2)
        waiters = [asyncio.create_task(result.get()) for _ in range(3)]
        await asyncio.sleep(0)
        await result.put("done")
        self.assertEqual(await asyncio.wait_for(asyncio.gather(*waiters), 0.1),
                         ["done"] * 3)

    async def test_failure_before_and_after_wait(self):
        for early in (True, False):
            result = AsyncResult(3)
            task = asyncio.create_task(result.get())
            if not early:
                await asyncio.sleep(0)
            await result.fail(RuntimeError("GPU worker failed"))
            with self.assertRaisesRegex(RuntimeError, "GPU worker failed"):
                await asyncio.wait_for(task, 0.1)

    async def test_cancelled_waiter_does_not_cancel_request(self):
        result = AsyncResult(4)
        task = asyncio.create_task(result.get())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await result.put("done")
        self.assertEqual(await result.get(), "done")


class RequestStatsTest(unittest.TestCase):
    def test_three_tokens_have_two_intervals(self):
        stat = SloStat(1, 1000, 1000, 1050, [1000, 1020, 1050])
        stat.post_process()
        self.assertEqual(stat.t_tokens, [.02, .03])
        self.assertEqual(len(stat.t_token_timestamps), 3)
        self.assertEqual(stat.t_decode, .05)

    def test_one_token_has_no_itl_and_counts_as_one_token(self):
        stat = SloStat(2, 1000, 1000, 1000, [1000])
        stat.post_process()
        self.assertEqual(stat.t_tokens, [])
        self.assertEqual(stat.t_token_timestamps, [1.0])
        self.assertEqual(stat.t_decode, 0.0)


class CapacityTest(unittest.TestCase):
    def test_budget_includes_colocated_weights_and_runtime(self):
        gib = 1024 ** 3
        # 16 GiB resident; 18 GiB budget leaves 2 GiB for the full-layer KV pool.
        self.assertEqual(kv_cache_block_budget(24 * gib, 8 * gib, .75, 2**20), 2048)

    def test_empty_budget_fails_with_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "No room for KV cache"):
            kv_cache_block_budget(24 * 1024**3, 8 * 1024**3, .6, 2**20)

    def test_invalid_utilization(self):
        for utilization in (0, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                kv_cache_block_budget(100, 80, utilization, 1)

    def test_request_capacity_matches_scheduler_rounding(self):
        validate_request_lengths(17, 17, 4096, 16, 4)
        with self.assertRaisesRegex(ValueError, "needs 4 KV blocks"):
            validate_request_lengths(17, 17, 4096, 16, 3)

    def test_rejects_context_overflow_and_empty_output(self):
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            validate_request_lengths(4090, 10, 4096, 16)
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_request_lengths(16, 0, 4096, 16)


if __name__ == "__main__":
    unittest.main()
