"""Intersect a worker's CUDA gaps, runnable queues and claimed batches on the Nsight clock."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from statistics import median


NS = 1_000_000_000
WAKE_SOURCES = {"top8", "top8_finalize", "expert_input", "attention_input"}


def unite(ranges):
    result = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def length(ranges):
    return sum(end - start for start, end in ranges)


def gaps(ranges, lo, hi):
    previous = lo
    empty = []
    for start, end in ranges:
        if start > previous:
            empty.append((previous, start))
        previous = end
    if previous < hi:
        empty.append((previous, hi))
    return empty


def load_markers(db, mask='WG.*', bounds=None):
    query = """
        SELECT n.start, n.end, COALESCE(n.text, s.value)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
        WHERE COALESCE(n.text, s.value) GLOB ?
    """
    args = [mask]
    if bounds:
        query += ' AND n.start < ? AND (n.start >= ? OR n.end >= ?)'
        args.extend((bounds[1], bounds[0], bounds[0]))
    rows = db.execute(query + ' ORDER BY n.start,n.rowid', args)
    markers = defaultdict(list)
    for start, end, name in rows:
        kind, _, value = name[3:].partition("=")
        if kind in ("ready", "claim", "work", "retire", "clock", "finalize"):
            value = int(value)
        markers[kind].append((start, end, value))
    return markers


def interval_on_trace(markers, origin, window):
    # perf_counter_ns on the same Linux node is system-wide. Sample the worker
    # counter inside the NVTX trace; neither a process start nor wall time is used.
    pairs = sorted((time_ns, ts) for ts, _, time_ns in markers["clock"])
    counter = [x for x, _ in pairs]
    trace = [t for _, t in pairs]
    if len(counter) < 2 or any(b <= a for a, b in zip(counter, counter[1:])) or any(
        b <= a for a, b in zip(trace, trace[1:])
    ):
        raise ValueError("Need at least two strictly increasing WG.clock anchors")
    def convert(target):
        if not counter[0] <= target <= counter[-1]:
            raise ValueError("Capture clock anchors do not enclose the requested window")
        i = min(max(bisect_right(counter, target) - 1, 0), len(counter) - 2)
        return trace[i] + round((target - counter[i]) * (trace[i+1] - trace[i])
                                / (counter[i+1] - counter[i]))
    lo, hi = (convert(origin + round(t * NS)) for t in window)
    offsets = [ts - x for x, ts in pairs]
    return lo, hi, {"anchors": len(pairs), "counter_to_trace_median_offset_ns": median(offsets),
                    "offset_range_ns": max(offsets) - min(offsets),
                    "method": "Piecewise linear interpolation of worker CLOCK_MONOTONIC/NVTX anchor pairs"}


def gpu_activity(db, lo, hi, device):
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kernel = "CUPTI_ACTIVITY_KIND_KERNEL"
    if kernel not in tables:
        kernel = "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"
    if kernel not in tables:
        raise ValueError("No CUDA kernel activity table; capture with CUDA graph node tracing")
    known = [kernel] + ["CUPTI_ACTIVITY_KIND_" + k for k in ("MEMCPY", "MEMCPY2", "MEMSET", "MEM_DECOMPRESS")]
    ids = set()
    for table in known:
        if table in tables:
            ids.update(r[0] for r in db.execute(
                f"SELECT DISTINCT deviceId FROM {table} WHERE start < ? AND end > ?", (hi, lo)))
    if device is None:
        if len(ids) != 1:
            raise ValueError(f"Expected one worker GPU, found {sorted(ids)}; specify --device")
        device = next(iter(ids))
    if device not in ids:
        raise ValueError(f"No hardware activity on device {device} in the requested window")
    # Classify names by ID once. CUTLASS demangled names can span kilobytes;
    # joining and re-decoding the full name for every kernel dominates runtime.
    nccl_names = {row[0] for row in db.execute("SELECT id FROM StringIds WHERE value LIKE '%nccl%'")}
    groups = defaultdict(list)
    for table in known:
        if table not in tables:
            continue
        query = f"SELECT a.start,a.end FROM {table} a"
        if table == kernel:
            query = f"SELECT a.start,a.end,a.demangledName FROM {table} a"
        counts = db.execute(query + " WHERE a.deviceId=? AND a.start<? AND a.end>?", (device, hi, lo))
        for start, end, *name in counts:
            if name:
                category = "nccl_kernel" if name[0] in nccl_names else "other_kernel"
            else:
                category = "memory"
            groups[category].append((max(start, lo), min(end, hi)))
    by_type = {k: unite(values) for k, values in groups.items()}
    active = unite(piece for ranges in by_type.values() for piece in ranges)
    return gaps(active, lo, hi), {
        "device": device, "active_ns": length(active),
        "union_by_kind_ns": {k: length(v) for k, v in by_type.items()},
        "scope": "CUDA kernels (including NCCL), memcopies and memsets on all recorded streams of this worker/device; groups may overlap."}


def unique_by_id(rows, name):
    result = {}
    for start, end, batch in rows:
        if batch in result:
            raise ValueError(f"Duplicate WG.{name} for batch {batch}")
        result[batch] = (start, end)
    return result


def held_intervals(markers, lo, hi):
    claims = unique_by_id(markers["claim"], "claim")
    work = unique_by_id(markers["work"], "work")
    retires = unique_by_id(markers["retire"], "retire")
    finalizes = unique_by_id(markers["finalize"], "finalize")
    pre, unretired, ambiguous = [], [], []
    checked = []
    all_ids = sorted(claims.keys() | work.keys() | retires.keys())
    for batch in all_ids:
        start = claims.get(batch, (None, None))[0]
        begin_work, end_work = work.get(batch, (None, None))
        retire = retires.get(batch, (None, None))[0]
        observed = [x for x in (start, begin_work, end_work, retire) if x is not None]
        if any(a > b for a,b in zip(observed, observed[1:])):
            raise ValueError(f"Invalid claim/work/retire order for batch {batch}")
        if (retire is not None and retire <= lo) or (start is not None and start >= hi):
            continue
        if None in (start, begin_work, end_work, retire):
            # Truncated edges cannot silently become idle/no-claim time.
            ambiguous.append((lo if start is None else start, hi if retire is None else retire))
            continue
        pre.append((start, end_work))
        unretired.append((end_work, retire))
        checked.append(batch)
    window_ids = sorted(b for b, (t, _) in claims.items() if lo <= t < hi)
    if not window_ids or window_ids[-1] - window_ids[0] + 1 != len(window_ids):
        raise ValueError("Missing or non-contiguous WG.claim IDs in measurement window")
    # A synthetic Finalize may execute asynchronously without a host-observed
    # completion fence.  A later non-Finalize completion on the same compute
    # stream is a conservative upper bound for work issued before its claim.
    completions = sorted((claims[b][0], ts) for b, (ts, _) in retires.items()
                         if b in claims and b not in finalizes)
    completion_claims = [x for x, _ in completions]
    guards = []
    for b in finalizes:
        if b not in retires:
            continue
        ts = retires[b][0]
        i = bisect_left(completion_claims, ts)
        guards.append((ts, completions[i][1] if i < len(completions) else hi))
    return {"pre": unite(pre), "submitted": unite(unretired),
            "finalize_guard": unite(guards), "unmatched": unite(ambiguous)}, {
        "claims_in_window": len(window_ids), "matched_intersecting_window": len(checked),
        "unmatched_intersecting_window": len(ambiguous),
        "finalize_retire_points_in_window": sum(lo <= t < hi for t, _ in finalizes.values())}


def queue_segments(markers, lo, hi):
    transitions = markers["ready"]
    wakes = markers["wake"]
    if any(state not in (0, 1) for _, _, state in transitions):
        raise ValueError("Ready flag must be 0 or 1")
    # WG.wake arrives directly after WG.ready=1 under the queue lock.
    # Capture-start snapshots might legitimately lack a wake.
    segments = []
    counts = Counter()
    times = [t for t, _, _ in wakes]
    state, start = None, lo
    for i, (ts, _, new) in enumerate(transitions):
        if ts >= hi:
            break
        if new == state:
            continue
        reason = "not_applicable"
        if state == 0 and new == 1:
            j = bisect_left(times, ts)
            next_transition = transitions[i + 1][0] if i + 1 < len(transitions) else float("inf")
            reason = "unknown"
            if j < len(wakes) and times[j] < next_transition:
                reason = wakes[j][2].split(":")[0]
                if reason not in WAKE_SOURCES:
                    raise ValueError(f"Unrecognized wake source: {wakes[j][2]}")
            if lo <= ts < hi:
                counts[reason] += 1
        if start < ts:
            segments.append((start, ts, state, reason))
        state, start = new, max(ts, lo)
    if start < hi:
        segments.append((start, hi, state, "right_censored" if state == 0 else "not_applicable"))
    return segments, counts


def partition(idle, markers, lo, hi, window_start_s=0):
    held, matches = held_intervals(markers, lo, hi)
    queues, wake_counts = queue_segments(markers, lo, hi)
    events = defaultdict(list)
    totals, wake_ns = Counter(), Counter()
    for name, pieces in dict(held, idle=idle).items():
        for start, end in pieces:
            start, end = max(lo, start), min(hi, end)
            if start < end:
                events[start].append((name, True))
                events[end].append((name, False))
    for start, _, state, wake in queues:
        events[start].append(("queue", (state, wake)))
    # Boundaries participate in the same sweep as state changes, so no idle
    # interval gets charged to the wrong bin when it crosses a boundary.
    edges = sorted({lo + (hi-lo)*i//6 for i in range(7)})
    bins = []
    for left, right in zip(edges, edges[1:]):
        events[left]
        bins.append(dict(start_s=window_start_s+(left-lo)/NS, duration_ns=right-left,
                         idle_ns=0, idle_queue_nonempty_ns=0, idle_queue_empty_ns=0,
                         idle_queue_unknown_ns=0, idle_empty_no_claim_ns=0))
    events[hi]
    flags = dict(idle=False, pre=False, submitted=False, finalize_guard=False, unmatched=False)
    queue, wake = None, "unknown"
    times = sorted(events)
    # Keep each largest complete hardware gap intact, even when several CPU/queue
    # states change inside it.
    largest = sorted(sorted(idle,key=lambda pair: pair[1]-pair[0], reverse=True)[:10])
    example_starts = [s for s, _ in largest]
    examples = [{"start_s": window_start_s + (s-lo)/NS, "end_s": window_start_s + (e-lo)/NS,
                 "duration_us": (e-s)/1000, "state_ns": Counter(), "empty_ends_by": set()}
                for s,e in largest]
    for t, t2 in zip(times, times[1:]):
        for name, value in events[t]:
            if name == "queue":
                queue, wake = value
            else:
                flags[name] = value
        if not flags["idle"]:
            continue
        q = {None: "queue_unknown", 0: "queue_empty", 1: "queue_nonempty"}[queue]
        if flags["unmatched"]:
            h = "marker_incomplete"
        elif flags["pre"]:
            h = "claimed_host_issuing"
        elif flags["submitted"]:
            h = "issued_before_host_retire"
        elif flags["finalize_guard"]:
            h = "possible_async_finalize"
        else:
            h = "no_observed_claim"
        totals[q, h] += t2 - t
        bucket = bins[bisect_right(edges, t) - 1]
        bucket["idle_ns"] += t2 - t
        bucket[f"idle_{q}_ns"] += t2 - t
        if q == "queue_empty" and h == "no_observed_claim":
            bucket["idle_empty_no_claim_ns"] += t2 - t
        ix = bisect_right(example_starts, t) - 1
        if ix >= 0 and t < largest[ix][1]:
            examples[ix]["state_ns"][f"{q}/{h}"] += t2-t
            if q == "queue_empty":
                examples[ix]["empty_ends_by"].add(wake)
        if (q, h) == ("queue_empty", "no_observed_claim"):
            wake_ns[wake] += t2 - t
    total = length(idle)
    if sum(totals.values()) != total:
        raise RuntimeError("Idle partition does not cover the observed CUDA gap duration")
    confirmed = totals["queue_empty", "no_observed_claim"]
    for sample in examples:
        sample["empty_ends_by"] = sorted(sample["empty_ends_by"])
    queue_empty = sum(dt for (q,_),dt in totals.items() if q == "queue_empty")
    upper = sum(dt for (q,h),dt in totals.items() if q != "queue_nonempty" and h != "claimed_host_issuing")
    return {
        "idle_ns": total, "idle_fraction_of_window": total/(hi-lo),
        "time_bins": bins,
        "queue_empty_without_observed_claim_ns": confirmed,
        "queue_empty_without_observed_claim_fraction_of_idle": confirmed/total if total else None,
        "possible_no_work_upper_ns": upper,
        "possible_no_work_upper_fraction_of_idle": upper/total if total else None,
        "queue_empty_during_gpu_idle_ns": queue_empty,
        "queue_empty_wall_ns": sum(end-start for start,end,q,_ in queues if q==0),
        "idle_partition_ns": {q: {h: dt for (q2,h),dt in totals.items() if q2 == q}
                              for q in ("queue_empty","queue_nonempty","queue_unknown")},
        "empty_spell_wakeups": dict(wake_counts),
        "queue_empty_without_observed_claim_ns_by_next_wake": dict(wake_ns),
        "largest_idle_gaps": sorted(examples,key=lambda sample: -sample["duration_us"]),
        **matches}


def analyze(database, choices, window, device):
    run = json.loads(choices.read_text())
    origin = run["start_ns"]
    if not 0 <= window[0] < window[1] <= (run["end_ns"] - origin)/NS:
        raise ValueError("Analysis window outside benchmark")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        clock_marks = load_markers(db, "WG.clock=*")
        lo, hi, clock = interval_on_trace(clock_marks, origin, window)
        markers = load_markers(db, bounds=(lo-2*NS, hi+2*NS))
        # The overlap loop retires non-finalize batches FIFO. One entire real
        # batch retired before lo bounds any older must-observe completion.
        # Fall back to all captured markers when our padding cannot prove that.
        prior_claims = {b for t,_,b in markers["claim"] if t < lo}
        prior_retired = {b for t,_,b in markers["retire"] if t <= lo}
        finalized = {b for _,_,b in markers["finalize"]}
        if not (prior_claims & prior_retired) - finalized or not any(
                t <= lo for t,_,_ in markers["ready"]):
            markers = load_markers(db)
        for kind in ("ready", "claim", "work", "retire"):
            if not markers[kind]:
                raise ValueError(f"No WG.{kind} markers in capture")
        idle, activity = gpu_activity(db, lo, hi, device)
    result = partition(idle, markers, lo, hi, window[0])
    if result["idle_ns"] + activity["active_ns"] != hi-lo:
        raise RuntimeError("CUDA active/idle coverage failed")
    return {"schema": "streaminfer.worker_gaps.v1", "window_s": window, "trace_window_ns": [lo,hi],
            "window_duration_ns": hi-lo, "clock": clock, "gpu": activity, **result,
            "interpretation": [
                "Ready means a locally runnable attention, expert or finalize queue.",
                "Host issuing covers selection, dequeuing, preprocessing and CUDA submission.",
                "Largest gap times are relative to the measured arrivals origin, interpolated on the trace clock.",
                "A submitted batch may have already finished on GPU before its normal host retirement.",
                "Finalize with no event is conservatively held until a later batch on the compute stream retires.",
                "Empty queues with no observed claim describe worker state; transport, CPU progress and missing branches can all delay the next ready work.",
                "Wake causes identify the next enqueue, not the cause of the entire preceding empty interval."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--choices", required=True, type=Path, help="Driver choices/run.json")
    parser.add_argument("--window", default="95,125", help="Half-open seconds from first measured submission")
    parser.add_argument("--device", type=int, help="Nsight CUDA activity device ID (auto for one GPU)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    window = list(map(float, args.window.split(",")))
    if len(window) != 2:
        parser.error("Expected --window START,END")
    outcome = analyze(args.sqlite, args.choices, window, args.device)
    args.output.write_text(json.dumps(outcome, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
