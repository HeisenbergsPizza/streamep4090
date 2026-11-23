"""Measure the SGLang HTTP endpoint with equally spaced, open-loop arrivals."""
import argparse
import asyncio
from bisect import bisect_left
import json
import math
import os
from pathlib import Path
import time

from profile_client_dpa4_ep4_b512 import (
    BASE_URL, fixed_prompt_ids, graph_disabled_from_env, validate_server,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=12.0, help="Global requests per second")
    parser.add_argument("--num-requests", type=int, default=1536)
    parser.add_argument("--decode-dummy", action="store_true",
                        help="Use the fake-KV decode server and exclude its placeholder prefill token")
    parser.add_argument("--window", metavar="START,END",
                        help="Window in seconds from start, default: full offered interval")
    parser.add_argument("--output", required=True, type=Path, help="New report JSON")
    args = parser.parse_args(argv)
    if args.num_requests <= 0 or not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("num-requests and rate must be positive and finite")
    offered_s = args.num_requests / args.rate
    try:
        start, end = map(float, args.window.split(",")) if args.window else (0.0, offered_s)
        if not 0 <= start < end <= offered_s:
            raise ValueError
    except ValueError:
        parser.error(f"--window must be START,END within [0,{offered_s:g}] s")
    args.window = (start, end)
    return args


async def read_generation(lines, input_count, output_count, now):
    """SGLang emits cumulative counts, which may repeat or jump across SSE events."""
    received = []
    cached = None
    async for line in lines:
        stamp = now()
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            if len(received) != output_count:
                raise ValueError(f"Received {len(received)} of {output_count} output tokens")
            return received, stamp, cached
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"SGLang stream error: {event['error']}")
        meta = event["meta_info"]
        if meta["prompt_tokens"] != input_count:
            raise ValueError(f"Unexpected server prompt length: {meta['prompt_tokens']}")
        count = meta["completion_tokens"]
        if type(count) is not int or not len(received) <= count <= output_count:
            raise ValueError(f"Invalid cumulative completion_tokens: {count}")
        if meta.get("cached_tokens") is not None:
            cached = meta["cached_tokens"]
        reason = meta.get("finish_reason")
        if isinstance(reason, dict) and reason.get("type") == "abort":
            raise RuntimeError(f"SGLang aborted the request: {reason}")
        received.extend([stamp] * (count - len(received)))
    raise RuntimeError("SGLang stream ended before [DONE]")


def percentiles(values):
    if not values:
        return {"mean": None, "p50": None, "p99": None}
    values = sorted(values)
    def at(q):
        i = q * (len(values) - 1)
        left = int(i)
        return values[left] + (i - left) * (values[math.ceil(i)] - values[left])
    return {"mean": sum(values) / len(values), "p50": at(.5), "p99": at(.99)}


def latencies(rows):
    return {
        "requests": len(rows),
        "e2e_ms": percentiles([(r["response_s"] - r["submitted_s"]) * 1000 for r in rows]),
        "ttft_ms": percentiles([(r["token_times_s"][0] - r["submitted_s"]) * 1000 for r in rows]),
        "tpot_ms": percentiles([(r["token_times_s"][-1] - r["token_times_s"][0]) * 1000
                                / (len(r["token_times_s"]) - 1) for r in rows if len(r["token_times_s"]) > 1]),
    }


def summarize(rows, duration, window):
    submitted = sorted(r["submitted_s"] for r in rows)
    finished = sorted(r["response_s"] for r in rows)
    tokens = sorted(t for r in rows for t in r["token_times_s"])

    def counts(start, end):
        # Half-open event counts; in-flight is the value just before each edge.
        a0, a1 = bisect_left(submitted, start), bisect_left(submitted, end)
        f0, f1 = bisect_left(finished, start), bisect_left(finished, end)
        n = bisect_left(tokens, end) - bisect_left(tokens, start)
        return dict(start_s=start, end_s=end, submitted=a1-a0, completed=f1-f0, output_tokens=n,
                    outstanding_start=a0-f0, outstanding_end=a1-f1,
                    req_s=(f1-f0)/(end-start), output_tok_s=n/(end-start))

    start, end = window
    return {
        "whole": counts(0, duration),
        "window": counts(start, end),
        "all_requests": latencies(rows),
        "arrivals_in_window": latencies([r for r in rows if start <= r["submitted_s"] < end]),
        "submission_lag_ms": percentiles([(r["submitted_s"] - r["scheduled_s"]) * 1000 for r in rows]),
        "bins_5s": [counts(i, min(i + 5.0, duration)) for i in range(0, math.ceil(duration), 5)],
    }


async def run_workload(session, prompts, rate, output_count, *, first_bootstrap_room=None):
    started = time.perf_counter()
    skip_tokens = int(first_bootstrap_room is not None)
    api_output_count = output_count + skip_tokens

    def elapsed():
        return time.perf_counter() - started

    async def generate(index, prompt, scheduled):
        payload = {"input_ids": prompt, "stream": True, "sampling_params": {
            "temperature": 0, "max_new_tokens": api_output_count, "ignore_eos": True,
        }}
        if first_bootstrap_room is not None:
            payload.update(bootstrap_host="2.2.2.2", bootstrap_room=first_bootstrap_room + index)
        submitted = elapsed()
        async with session.post(BASE_URL + "/generate", json=payload) as response:
            response.raise_for_status()
            tokens, finished, cached = await read_generation(response.content, len(prompt),
                                                              api_output_count, elapsed)
        # read_generation expands each cumulative count increment to its SSE arrival
        # time. Drop exactly one token even if the first event delivers several.
        row = dict(index=index, scheduled_s=scheduled, submitted_s=submitted,
                   token_times_s=tokens[skip_tokens:], response_s=finished, cached_tokens=cached,
                   api_completion_tokens=len(tokens))
        if skip_tokens:
            row.update(bootstrap_room=payload["bootstrap_room"], bootstrap_token_s=tokens[0])
        return row

    # The producer does not wait for completions or reserve a fixed concurrency slot.
    async with asyncio.TaskGroup() as group:
        tasks = []
        for i, prompt in enumerate(prompts):
            scheduled = i / rate
            await asyncio.sleep(max(0, scheduled - elapsed()))
            tasks.append(group.create_task(generate(i, prompt, scheduled)))
    return [task.result() for task in tasks], elapsed()


def make_inputs(tokenizer, n, phase, length):
    prompts = [fixed_prompt_ids(tokenizer, i, phase, length) for i in range(n)]
    if any(len(p) != length for p in prompts) or len(set(map(tuple, prompts))) != n:
        raise ValueError("Prompt IDs must be distinct and have the configured length")
    return prompts


def check_decode_mode(info, decode_dummy):
    mode = info.get("disaggregation_mode")
    load_format = info.get("load_format")
    if decode_dummy:
        if (mode != "decode" or load_format != "dummy"
                or info.get("disaggregation_transfer_backend") != "fake"
                or info.get("speculative_algorithm") not in (None, "NONE")):
            raise ValueError("Use serve-decode-dummy for fake KV, dummy weights and ordinary decode")
    elif mode not in (None, "null") or load_format == "dummy":
        raise ValueError("Use ordinary serve mode for prefill plus decode, or pass --decode-dummy")


async def benchmark(args, report):
    import aiohttp
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"], local_files_only=True)
    warmup = make_inputs(tokenizer, 10, "预热", 100)
    prompts = make_inputs(tokenizer, args.num_requests, "测量", 64)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                     timeout=aiohttp.ClientTimeout(total=1800),
                                     trust_env=False) as session:
        async with session.get(BASE_URL + "/health") as response:
            response.raise_for_status()
        async with session.get(BASE_URL + "/server_info") as response:
            response.raise_for_status()
            info = await response.json()
        validate_server(info, disable_cuda_graph=graph_disabled_from_env(), require_worker_cap=False)
        check_decode_mode(info, args.decode_dummy)
        if info.get("disable_radix_cache") is not True:
            raise ValueError("Use the serve launcher with --disable-radix-cache for this workload")
        report["server_info"] = info
        warmup_rows, _ = await run_workload(session, warmup, 5.0, 50,
                                           first_bootstrap_room=1 if args.decode_dummy else None)
        report["warmup"]["api_completion_tokens"] = sum(r["api_completion_tokens"] for r in warmup_rows)
        report["warmup"]["measured_output_tokens"] = sum(len(r["token_times_s"]) for r in warmup_rows)
        print(f"Warmup complete. Sending {len(prompts)} requests at {args.rate:g} req/s.", flush=True)
        rows, duration = await run_workload(session, prompts, args.rate, 128,
                                           first_bootstrap_room=len(warmup) + 1 if args.decode_dummy else None)
    report["requests"] = rows
    report.update(summarize(rows, duration, args.window))
    report["api_completion_tokens"] = sum(r["api_completion_tokens"] for r in rows)
    if report["whole"]["completed"] != len(prompts) or report["whole"]["output_tokens"] != len(prompts) * 128:
        raise RuntimeError("Incomplete measured workload")
    print(json.dumps({"whole": report["whole"], "window": report["window"],
                      "arrivals_in_window": report["arrivals_in_window"]}, indent=2))


def main(argv=None):
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"format": "sglang.uniform_http.v1", "rate": args.rate, "num_requests": args.num_requests,
              "input_tokens": 64, "output_tokens_per_request": 128, "measurement_window_s": args.window,
              "mode": "decode-dummy" if args.decode_dummy else "serve",
              "excluded_bootstrap_tokens_per_request": int(args.decode_dummy),
              "warmup": {"requests": 10, "rate": 5, "input_tokens": 100, "output_tokens": 50},
              "status": "running"}
    # Open exclusively: preserve an earlier trial when a sweep is rerun.
    with args.output.open("x", encoding="utf-8") as f:
        try:
            asyncio.run(benchmark(args, report))
            report["status"] = "complete"
        except BaseException as exc:
            import traceback
            report.update(status="failed", error="".join(traceback.format_exception(exc)))
            raise
        finally:
            json.dump(report, f, ensure_ascii=False, allow_nan=False)
            f.write("\n")
    print(f"PASS: {args.output}")


if __name__ == "__main__":
    main()
