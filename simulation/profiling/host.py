"""Extract engine-thread phases from an exported Nsight Systems SQLite report."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from contextlib import closing
import csv
import re
from pathlib import Path
import sqlite3
from statistics import median

from .common import StageWriter, load_manifest, write_json


class Intervals:
    """Disjoint intervals with a prefix sum for fast nested-wait subtraction."""

    def __init__(self, spans):
        self.starts, self.ends = [], []
        for a, b in sorted(spans):
            if b <= a:
                continue
            if self.ends and a <= self.ends[-1]:
                self.ends[-1] = max(b, self.ends[-1])
            else:
                self.starts.append(a)
                self.ends.append(b)
        self.prefix = [0]
        for a, b in zip(self.starts, self.ends):
            self.prefix.append(self.prefix[-1] + b - a)

    def before(self, t):
        i = bisect_right(self.ends, t)
        amount = self.prefix[i]
        if i < len(self.starts):
            amount += max(0, t - self.starts[i])
        return amount

    def overlap(self, a, b):
        return self.before(b) - self.before(a)


def table_exists(db, name):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone())


_NVTX = "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id"
_LABEL = "COALESCE(n.text,s.value)"
_MAIN = {"Engine.preprocess": "pre", "Engine.postprocess": "post",
         "Engine.previous_event_wait": "previous_event_wait", "Scheduler::dequeue_merge_unified": "dequeue"}


def thread_id(db, requested):
    if not all(table_exists(db, t) for t in ("NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_RUNTIME")):
        raise ValueError("Export a CUDA+NVTX Nsight Systems SQLite trace with CUDA runtime API records.")
    tids = [r[0] for r in db.execute(f"SELECT DISTINCT n.globalTid {_NVTX} "
                                     f"WHERE {_LABEL}='Scheduler::dequeue_merge_unified'")]
    if requested is not None and requested not in tids:
        raise ValueError(f"No dequeue on engine globalTid {requested}; available: {tids}")
    if requested is None and len(tids) != 1:
        raise ValueError(f"Select one engine with --tid (Nsight globalTid); available: {tids}")
    return requested if requested is not None else tids[0]


def waits(db, tid, lo, hi):
    ranges = []
    def load(table, patterns):
        if not table_exists(db, table):
            return
        terms = " OR ".join("s.value GLOB ?" for _ in patterns)
        ranges.extend(db.execute(f"SELECT a.start,a.end FROM {table} a JOIN StringIds s ON s.id=a.nameId "
                                 f"WHERE a.globalTid=? AND a.start<? AND a.end>? AND ({terms})",
                                 (tid, hi, lo, *patterns)))
    load("CUPTI_ACTIVITY_KIND_RUNTIME", ["*Synchronize*", "cudaMemcpy_v*", "cudaMemcpy2D_v*"])
    load("OSRT_API", ["*cond*wait*", "*sem_wait*", "*sem_timedwait*", "*nanosleep*", "*futex*wait*"])
    # Sometimes the engine polls an event rather than calling cudaEventSynchronize.
    ranges.extend(db.execute(f"SELECT n.start,n.end {_NVTX} WHERE n.globalTid=? AND n.start<? AND n.end>? "
                             f"AND {_LABEL}='Engine.previous_event_wait'", (tid, hi, lo)))
    return Intervals(ranges)


def gather_launches(db, tid, lo, hi):
    """Resolve CPU runtime launches to eager GPU kernels by process + CUDA correlation."""
    kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if not table_exists(db, kernel_table):
        return []
    columns = {item[1] for item in db.execute(f"PRAGMA table_info({kernel_table})")}
    graph = " AND COALESCE(k.graphId,0)=0" if "graphId" in columns else ""
    # globalTid and globalPid encode the same process with a 24-bit thread field.
    pid = tid & ~0xffffff
    kernels = defaultdict(list)
    for corr, name, a, b, dev in db.execute(
        f"SELECT k.correlationId,s.value,k.start,k.end,k.deviceId FROM {kernel_table} k "
        "JOIN StringIds s ON k.demangledName=s.id WHERE k.globalPid=? "
        "AND (s.value LIKE '%gather_tokens_kernel%' OR s.value LIKE '%gather_and_sum_tokens_kernel%')" + graph,
        (pid,)):
        kind = "gather_and_sum" if "gather_and_sum_tokens_kernel" in name else "gather"
        kernels[corr].append((kind, a, b, dev))
    result = []
    for a, b, corr in db.execute("SELECT r.start,r.end,r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                                 "JOIN StringIds s ON r.nameId=s.id WHERE r.globalTid=? AND r.start>=? "
                                 "AND r.end<=? AND s.value LIKE '%LaunchKernel%'", (tid, lo, hi)):
        if corr in kernels and len(kernels[corr]) == 1:
            result.append((a, b, corr, *kernels[corr][0]))
    return sorted(result)


def samples(db, tid, lo, hi):
    masks = ("Engine.*", "Scheduler::dequeue_merge_unified", "WG.work=*", "WG.retire=*",
             "AttnExecutor.execute", "ExpertsExecutor.execute")
    cond = " OR ".join(f"{_LABEL} GLOB ?" for _ in masks)
    nvtx = db.execute(f"SELECT n.start,n.end,{_LABEL} {_NVTX} WHERE n.globalTid=? "
                      f"AND n.start>=? AND n.start<? AND ({cond}) ORDER BY n.start,n.rowid",
                      (tid, lo, hi, *masks))
    phases, hints, works, retires, schedule = [], [], [], [], []
    for a, b, label in nvtx:
        if label.startswith("WG.retire="):
            retires.append((a, int(label.split("=")[1])))
        if b is None or b > hi or b <= a:
            continue
        if label in _MAIN or label == "Engine.execute" or label.startswith("Engine.execute."):
            phases.append((a, b, _MAIN.get(label, "execute"), label))
        elif label == "Engine.schedule.sample_1_in_64":
            schedule.append((a, b))
        elif label.startswith("WG.work="):
            works.append((a, b, int(label.split("=")[1])))
        elif label in ("ExpertsExecutor.execute", "AttnExecutor.execute"):
            hints.append((a, b, "expert" if label.startswith("Experts") else "attention"))
    if not phases:
        raise ValueError("Window contains no complete engine phases. Window arguments are ns in SQLite time.")
    phases.sort()
    if any(b > next_a for (_, b, *_), (next_a, *_) in zip(phases, phases[1:])):
        raise ValueError("Overlapping top-level engine phases on the selected thread; inspect instrumentation.")
    work_starts = [r[0] for r in works]
    hint_starts = [r[0] for r in hints]
    retire_starts = [r[0] for r in retires]
    stages, work_stage = {}, {}
    for i, (a, b, phase, label) in enumerate(phases):
        if phase != "execute":
            continue
        if label.startswith("Engine.execute.attn."):
            stage = "attention"
        elif label.startswith("Engine.execute.expert."):
            stage = "expert"
        else:
            contained = {r[2] for r in hints[bisect_left(hint_starts, a):bisect_left(hint_starts, b)] if r[1] <= b}
            stage = contained.pop() if len(contained) == 1 else "unknown"
        match = re.search(r"\.batch=(\d+)$", label)
        stage_and_rows = (stage, int(match[1]) if match else None)
        stages[i] = stage_and_rows
        idx = bisect_right(work_starts, a) - 1
        if idx >= 0 and b <= works[idx][1]:
            work_stage[works[idx][2]] = stage_and_rows

    waiting = waits(db, tid, lo, hi)
    launches = gather_launches(db, tid, lo, hi)
    launch_starts = [r[0] for r in launches]
    def record(a, b, phase, stage_data=("unknown", None), kind=None, gpu=None, submit_us=None):
        blocked = waiting.overlap(a, b)
        return {"phase": phase, "stage": stage_data[0], "batch_rows": stage_data[1],
                "kind": kind or "", "start_ns": a, "end_ns": b,
                "wall_us": (b-a)/1000, "known_wait_us": blocked/1000,
                "residual_wall_us": (b-a-blocked)/1000,
                "sampled_schedule_to_submit_residual_us": submit_us,
                "correlation_id": gpu[2] if gpu else "", "gpu_start_ns": gpu[4] if gpu else "",
                "gpu_end_ns": gpu[5] if gpu else "", "gpu_device": gpu[6] if gpu else ""}

    sampled_prefix = {}
    phase_starts = [r[0] for r in phases]
    for a, b in schedule:
        first = bisect_left(phase_starts, a)
        inside = phases[first:bisect_left(phase_starts, b)]
        if first and phases[first - 1][1] > a:
            continue
        if not inside:
            # Could be an empty poll or a path without dequeue instrumentation.
            yield record(a, b, "sampled_schedule_no_observed_dequeue")
        elif len(inside) == 1 and inside[0][2] == "dequeue" and inside[0][1] <= b:
            sampled_prefix[inside[0][0]] = a

    for i, (a, b, phase, label) in enumerate(phases):
        if phase == "dequeue":
            # A sampled outer schedule call containing *this* dequeue selected work.
            # Its prefix includes the Python entry and native selection. Count only
            # the prefix, so the nested dequeue never overlaps these CPU samples.
            stage_data = ("unknown", None)
            if i + 1 < len(phases) and phases[i + 1][2] == "pre":
                stage_data = stages.get(i + 2, stage_data)
            prefix = sampled_prefix.get(a)
            if prefix is not None and prefix < a:
                yield record(prefix, a, "sampled_selection_to_dequeue", stage_data)
            candidates = [r for r in launches[bisect_left(launch_starts, a):bisect_left(launch_starts, b)] if r[1] <= b]
            if len(candidates) == 1:
                gpu = candidates[0]
                x, y, _, kind, *_ = gpu
                yield record(a, x, "dequeue_before_launch", stage_data, kind=kind)
                submit_us = (y-prefix-waiting.overlap(prefix, y))/1000 if prefix is not None else None
                yield record(x, y, "gather_launch_api", stage_data, kind=kind, gpu=gpu, submit_us=submit_us)
                yield record(y, b, "dequeue_after_launch", stage_data, kind=kind)
            else:
                # May be a bypass, missing CUDA data, or multiple launches; never infer which.
                yield record(a, b, "dequeue_unresolved", stage_data)
            continue
        stage = ("unknown", None)
        if phase == "pre" and i + 1 in stages:
            stage = stages[i + 1]
        if phase == "post":
            idx = bisect_right(retire_starts, a) - 1
            if idx >= 0 and i > 0 and retires[idx][0] >= phases[i - 1][1]:
                stage = work_stage.get(retires[idx][1], ("unknown", None))
        if phase == "execute":
            stage = stages[i]
        yield record(a, b, phase, stage)


def run(args):
    load_manifest(args.bundle)
    if not args.sqlite.is_file():
        raise ValueError(f"Missing Nsight Systems SQLite export: {args.sqlite}")
    lo, hi = args.start_ns, args.end_ns
    if not 0 <= lo < hi:
        raise ValueError("Require 0 <= --start-ns < --end-ns, in SQLite trace time.")
    with closing(sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        tid = thread_id(db, args.tid)
        with StageWriter(args.bundle, "host") as output:
            groups = defaultdict(list)
            sampled_submit = []
            with (output / "raw.csv").open("w", newline="") as f:
                writer = None
                for row in samples(db, tid, lo, hi):
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    key = (row["phase"], row["stage"], row["kind"])
                    groups[key].append(row["residual_wall_us"])
                    if row["sampled_schedule_to_submit_residual_us"] is not None:
                        sampled_submit.append(row["sampled_schedule_to_submit_residual_us"])
            params, summaries = {}, []
            for (phase, stage, kind), values in sorted(groups.items()):
                mid = median(values)
                summaries.append(dict(phase=phase, stage=stage, kind=kind, n=len(values),
                                      median_residual_wall_us=mid))
                if args.use_wall_proxy:
                    if phase in ("pre", "post") and stage in ("expert", "attention"):
                        params[f"--cpu-{phase}-{stage}-us"] = mid
                    elif phase == "dequeue_after_launch":
                        name = "reduce" if kind == "gather_and_sum" else "gather"
                        params[f"--cpu-{name}-tail-us"] = mid
            if args.use_wall_proxy and len(sampled_submit) >= 5:
                # This observed whole span already includes queue traversal and
                # row/fragment processing at the sampled workload mix.
                params.update({"--cpu-fixed-us": median(sampled_submit), "--cpu-per-row-us": 0.0,
                               "--cpu-per-fragment-us": 0.0, "--cpu-per-queue-us": 0.0})
            write_json(output / "parameters.json", params)
            write_json(output / "details.json", dict(
                source=str(args.sqlite.resolve()), source_bytes=args.sqlite.stat().st_size,
                source_device_identity="caller-supplied exported trace; match its hardware and configuration to the manifest",
                engine_global_tid=tid, trace_window_ns=[lo, hi],
                use_wall_proxy=args.use_wall_proxy, osrt_table=table_exists(db, "OSRT_API"),
                sampled_selection_to_submit={"n": len(sampled_submit),
                        "median_us": median(sampled_submit) if sampled_submit else None,
                        "basis": "Positive sampled Python schedule through matched CUDA launch return, "
                                 "less recognized waits. Overlaps constituent CSV phases; never add to them."},
                groups=summaries, measurements="Complete ranges on one engine thread only. Units are microseconds.",
                scope="Residual wall subtracts the union of recognized blocking CUDA/OS calls and event waits. "
                      "It can still include driver blocking, lock contention and descheduling. "
                      "Preprocess uses the immediately following execute stage; postprocess uses WG.retire/work IDs. "
                      "Unlabeled stages stay unknown. Sampled scheduler ranges without a dequeue and GPU durations are excluded from CPU costs.",
                calibration="With --use-wall-proxy only, fixed pre/post and gather-tail medians approximate "
                            "effective engine service at this workload. Keep per-row pre/post defaults at zero "
                            "when using these estimates. With >=5 attributable sampled submissions, use their median as a single fixed cost with zero row, fragment and queue slopes. It represents that workload mix, including wrapper entry and submission. Independent selection complexity and model launches remain unfit."))


def add_commands(commands):
    p = commands.add_parser("host", help="Extract engine timing samples from an Nsight Systems SQLite export")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--sqlite", type=Path, required=True)
    p.add_argument("--tid", type=int, help="Nsight engine globalTid; detected when unique")
    p.add_argument("--start-ns", type=int, default=0, help="Start in SQLite trace nanoseconds")
    p.add_argument("--end-ns", type=int, default=(2**63-1), help="End in SQLite trace nanoseconds")
    p.add_argument("--use-wall-proxy", action="store_true", help="Export residual host wall medians as "
                   "approximate fixed scheduling/pre/post/tail parameters. Unknown phases stay uncalibrated.")
    p.set_defaults(func=run)
