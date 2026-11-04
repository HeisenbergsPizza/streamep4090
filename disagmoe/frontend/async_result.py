"""Request completion shared by the controller and its asyncio callers."""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from disagmoe.frontend.datatypes import SloStat


class AsyncResult:
    def __init__(self, req_id: int):
        self.req_id = req_id
        self._finished = asyncio.Event()
        self.slo_stat = None
        self._error = None

    async def wait(self):
        # Event remembers a completion that arrives before get()/wait().
        await self._finished.wait()
        if self._error is not None:
            raise self._error

    async def get(self) -> "SloStat":
        await self.wait()
        return self.slo_stat

    async def put(self, slo_stat: "SloStat"):
        if not self._finished.is_set():
            self.slo_stat = slo_stat
            self._finished.set()

    async def fail(self, error: Exception):
        if not self._finished.is_set():
            self._error = error
            self._finished.set()
