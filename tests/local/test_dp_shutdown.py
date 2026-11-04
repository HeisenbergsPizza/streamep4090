"""Exercise scheduler shutdown without CUDA, including capacity exhaustion."""

import asyncio
import sys
import typing
import unittest
from unittest.mock import patch

# Production uses Python 3.12. This annotation-only shim lets a CPU developer
# run these tests on Python 3.10/3.11 as well, without importing the GPU stack.
if sys.version_info >= (3, 10):
    with patch.object(typing, "override", getattr(typing, "override", lambda fn: fn), create=True):
        from disagmoe.scheduler.dpscheduler import DPSchedulerMax


@unittest.skipIf(sys.version_info < (3, 10), "scheduler annotations require Python >=3.10")
class ShutdownTest(unittest.IsolatedAsyncioTestCase):
    async def test_stop_while_idle_leaves_no_queue_waiter(self):
        scheduler = DPSchedulerMax(1, 16)
        scheduler.start({0: 4})
        await asyncio.sleep(0.01)
        await asyncio.wait_for(scheduler.terminate(), .2)
        self.assertTrue(scheduler._loop_task.done())
        self.assertFalse(scheduler.waiting_queue._getters)

    async def test_stop_while_waiting_for_capacity(self):
        scheduler = DPSchedulerMax(1, 16)
        submitted = []
        scheduler.start({0: 1})
        scheduler.put_request(lambda *args: submitted.append(args), 1, 32, 16, 16)
        await asyncio.sleep(.01)
        self.assertFalse(submitted)
        await asyncio.wait_for(scheduler.terminate(), .2)
        self.assertFalse(submitted)

    async def test_release_capacity_still_dispatches_next_request(self):
        scheduler = DPSchedulerMax(1, 16)
        submitted = []
        scheduler.start({0: 2})
        scheduler.put_request(lambda *args: submitted.append(args), 1, 32, 16, 16)
        scheduler.put_request(lambda *args: submitted.append(args), 2, 32, 16, 16)
        await asyncio.sleep(.01)
        self.assertEqual([args[0] for args in submitted], [1])
        scheduler.del_seq(1)
        await asyncio.sleep(.01)
        self.assertEqual([args[0] for args in submitted], [1, 2])
        scheduler.del_seq(2)
        await asyncio.wait_for(scheduler.terminate(), .2)


if __name__ == "__main__":
    unittest.main()
