"""Single-node open-loop decode arrivals and monotonic observation windows."""

import asyncio
import time


async def run_paced_requests(master, args):
    """Submit to the admission queue at absolute deadlines, independent of completion."""
    started = time.perf_counter()
    tasks = []

    async def collect(response, index, scheduled, submitted):
        result = await response.get()
        received = time.perf_counter()
        tokens = result.t_token_timestamps
        if result.req_id != response.req_id or not tokens or len(tokens) != args.min_output_len:
            raise ValueError(f"Incomplete paced decode result for request {response.req_id}")
        if (response.admitted_at is None or not response.admitted_at <= tokens[0] <= tokens[-1] <= received
                or any(t1 < t0 for t0, t1 in zip(tokens, tokens[1:]))):
            raise ValueError(f"Inconsistent paced decode clock for request {response.req_id}")
        row = dict(req_id=response.req_id, snapshot_index=index,
                   scheduled_s=scheduled, submitted_s=submitted,
                   admitted_s=response.admitted_at - started,
                   token_times_s=[t - started for t in tokens],
                   response_s=received - started)
        return result, row

    try:
        for index in range(args.num_requests):
            scheduled = index / args.rate
            delay = (started + scheduled) - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            submitted = time.perf_counter() - started
            response = master.put_snapshot_request(index, args.min_input_len, args.min_output_len)
            tasks.append(asyncio.create_task(collect(response, index, scheduled, submitted)))
        observations = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    results, rows = zip(*observations)
    duration = max(row["response_s"] for row in rows)
    window = args.arrival_window or (0.0, args.num_requests / args.rate)
    report = dict(offered_rate=args.rate, arrival_pattern="constant", duration_s=duration,
                  window=summarize_window(rows, *window), requests=rows)
    return results, report


def summarize_window(rows, start, end):
    """Count events in [start,end), with outstanding work immediately before each edge."""
    if not 0 <= start < end:
        raise ValueError("Measurement window must have positive duration")

    def before(key, edge):
        return sum(row[key] < edge for row in rows)

    arrivals = before("scheduled_s", end) - before("scheduled_s", start)
    completions = before("response_s", end) - before("response_s", start)
    tokens = sum(start <= t < end for row in rows for t in row["token_times_s"])
    return dict(
        start_s=start, end_s=end, scheduled_requests=arrivals,
        completed_requests=completions, output_tokens=tokens,
        req_throughput=completions / (end - start),
        token_throughput=tokens / (end - start),
        outstanding_start=before("scheduled_s", start) - before("response_s", start),
        outstanding_end=before("scheduled_s", end) - before("response_s", end),
        waiting_admission_start=before("submitted_s", start) - before("admitted_s", start),
        waiting_admission_end=before("submitted_s", end) - before("admitted_s", end),
    )
