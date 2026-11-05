"""Observation-only windows for open-loop decode workloads.

No sampling events or random draws are introduced. State integrals use the
piecewise-constant simulator state before its next event. Window event counts
are half-open; boundary states are left limits for conservation checks.
"""
from __future__ import annotations

from collections import Counter, deque
import csv
import math
from pathlib import Path


_EVENTS = ("arrived_requests", "admitted_requests", "completed_requests", "completed_tokens")
_METRICS = ("attention_ready_wait", "expert_ready_wait", "finalize_ready_wait",
            "network_receive_wait", "join_wait", "itl", "admission_wait",
            "request_latency", "token_latency")
_QUANTILES = {"itl", "admission_wait", "request_latency", "token_latency"}
_STAGES = ("attention", "expert", "finalize")
_CHANNELS = ("intra", "inter")


class _Stats:
    def __init__(self, quantiles=False):
        self.count, self.total, self.maximum = 0, 0.0, 0.0
        self.values = [] if quantiles else None

    def add(self, values):
        if not values:
            return
        self.count += len(values)
        self.total += sum(values)
        self.maximum = max(self.maximum, max(values))
        if self.values is not None:
            self.values.extend(values)

    def summary(self):
        result = {"count": self.count, "mean": self.total / self.count if self.count else None,
                  "sum": self.total, "max": self.maximum if self.count else None}
        ordered = sorted(self.values) if self.values else []
        for percentile in (50, 90, 95, 99):
            if ordered:
                position = (len(ordered) - 1) * percentile / 100
                lo, hi = math.floor(position), math.ceil(position)
                value = ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
            else:
                value = None
            result[f"p{percentile}"] = value
        result["quantiles_exact"] = self.values is not None
        return result


class _Bucket:
    def __init__(self, start, end, num_gpus, detailed=False,
                 receive_mode="fixed_delay_no_contention"):
        self.start, self.end = start, end
        self.duration = 0.0
        self.integral = Counter()
        self.busy = [0.0] * num_gpus
        self.stage_busy = [dict.fromkeys(_STAGES, 0.0) for _ in range(num_gpus)]
        self.active_cap_full_ms = 0.0
        self.receive_mode = receive_mode
        self.receive_busy = ({channel: [0.0] * num_gpus for channel in _CHANNELS}
                             if receive_mode == "receiver_fifo" else None)
        self.receive_queue = ({channel: [0.0] * num_gpus for channel in _CHANNELS}
                              if self.receive_busy is not None else None)
        self.start_state = self.end_state = None
        self.last_state = None
        self.counts = dict.fromkeys(_EVENTS, 0)
        self.age_at_end = None
        self.detailed = detailed
        self.metrics = {key: _Stats(key in _QUANTILES) for key in _METRICS} if detailed else {}
        self.histograms = {key: Counter() for key in ("attention", "expert_group", "expert_individual", "finalize")}
        self.schedules = dict.fromkeys(("attention", "expert", "finalize"), 0)
        self.queue_switches = [0] * num_gpus
        self.layer_switches = [0] * num_gpus
        self.stage_switches = [0] * num_gpus

    def integrate(self, left, right, state, busy, stages, active_cap_full, receive):
        duration = max(0.0, min(right, self.end) - max(left, self.start))
        if duration <= 0:
            return
        self.duration += duration
        self.integral.update({key: value * duration for key, value in state.items()})
        for gpu, value in enumerate(busy):
            self.busy[gpu] += value * duration
            if stages[gpu] is not None:
                self.stage_busy[gpu][stages[gpu]] += duration
        if active_cap_full:
            self.active_cap_full_ms += duration
        if receive is not None:
            for channel, resources in zip(_CHANNELS, receive):
                channel_busy, channel_queue = self.receive_busy[channel], self.receive_queue[channel]
                for gpu, (count, queued) in enumerate(resources):
                    channel_busy[gpu] += count * duration
                    channel_queue[gpu] += queued * duration
        self.last_state = dict(state)

    def summary(self, observation_end, state_at_stop, *, right_closed=False):
        observed_end = min(self.end, observation_end)
        covered = self.duration > 0
        complete = observation_end >= self.end
        seconds = self.duration / 1000
        end_state = self.end_state if complete else self.last_state
        if right_closed or (covered and not complete):
            end_state = dict(state_at_stop)
        result = {
            "start_ms": self.start, "end_ms": self.end if complete else max(self.start, observed_end),
            "requested_end_ms": self.end,
            "observed_start_ms": self.start if covered or right_closed else None,
            "observed_end_ms": observed_end if covered or right_closed else None,
            "observed_duration_ms": self.duration, "coverage_complete": complete,
            "right_closed_at_observation_stop": right_closed,
            "counts": dict(self.counts),
            "rates": {"arrival_requests_per_second": self.counts["arrived_requests"] / seconds if seconds else None,
                      "admission_requests_per_second": self.counts["admitted_requests"] / seconds if seconds else None,
                      "completed_requests_per_second": self.counts["completed_requests"] / seconds if seconds else None,
                      "output_tokens_per_second": self.counts["completed_tokens"] / seconds if seconds else None},
            "state_mean": {key: value / self.duration for key, value in self.integral.items()} if covered else None,
            "state_at_start": self.start_state, "state_at_end": end_state,
            "ages_at_end_ms": self.age_at_end,
            "gpu_busy_ms": list(self.busy),
            "gpu_busy_fraction": [value / self.duration if covered else None for value in self.busy],
            "gpu_stage_busy_ms": [dict(values) for values in self.stage_busy],
            "gpu_stage_busy_fraction": [
                {stage: value / self.duration if covered else None for stage, value in values.items()}
                for values in self.stage_busy],
            "active_cap_full_ms": self.active_cap_full_ms,
            "active_cap_full_fraction": self.active_cap_full_ms / self.duration if covered else None,
            "receive_resource_mode": self.receive_mode,
            "receive_busy_ms": ({channel: list(values) for channel, values in self.receive_busy.items()}
                                if self.receive_busy is not None else None),
            "receive_busy_fraction": (
                {channel: [value / self.duration if covered else None for value in values]
                 for channel, values in self.receive_busy.items()} if self.receive_busy is not None else None),
            "receive_queue_message_ms": (
                {channel: list(values) for channel, values in self.receive_queue.items()}
                if self.receive_queue is not None else None),
            "receive_queue_messages_mean": (
                {channel: [value / self.duration if covered else None for value in values]
                 for channel, values in self.receive_queue.items()} if self.receive_queue is not None else None),
        }
        if self.detailed:
            result.update(
                metrics_ms={key: metric.summary() for key, metric in self.metrics.items()},
                schedule_counts=dict(self.schedules),
                batch_histograms={key: dict(hist) for key, hist in self.histograms.items()},
                batch_mean={key: sum(size * count for size, count in hist.items()) / sum(hist.values())
                            if hist else None for key, hist in self.histograms.items()},
                queue_switches_by_gpu=list(self.queue_switches),
                layer_index_switches_by_gpu=list(self.layer_switches),
                stage_switches_by_gpu=list(self.stage_switches),
            )
        return result


class SteadyMetricsObserver:
    """Collect named (possibly overlapping) windows and 100-ms state bins.

    Ready waits belong to their service-start window, ITL/token latency to the
    output-completion window, admission waits to admission, and request latency
    to request completion. Exact quantiles retain token/request-level samples;
    high-volume branch waits retain only exact count/sum/mean/max, without RNG.
    """

    # This observer consumes batch/resource identity but never candidate scores.
    needs_schedule_details = False

    def __init__(self, measurement_windows_ms, sample_interval_ms=100.0, request_log_path=None):
        if not math.isfinite(sample_interval_ms) or sample_interval_ms <= 0:
            raise ValueError("sample_interval_ms must be finite and positive")
        self.window_specs = []
        labels = set()
        for label, start, end in measurement_windows_ms:
            if not isinstance(label, str) or not label or label in labels:
                raise ValueError("measurement window labels must be unique nonempty strings")
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError("measurement windows require finite 0 <= start < end")
            labels.add(label)
            self.window_specs.append((label, float(start), float(end)))
        if not self.window_specs:
            raise ValueError("at least one measurement window is required")
        self.interval = float(sample_interval_ms)
        self.request_log_path = None if request_log_path is None else Path(request_log_path).expanduser().resolve()
        if self.request_log_path is not None and self.request_log_path.exists():
            raise FileExistsError(self.request_log_path)
        self.bound = False
        self._finished = None

    def bind(self, sim):
        if self.bound:
            raise ValueError("each simulation needs a fresh SteadyMetricsObserver")
        self.bound = True
        self.ticks_per_ms, self.num_gpus = sim.ticks_per_ms, sim.num_gpus
        self.active_cap = getattr(sim, "active_cap", None)
        self.receive_mode = ("receiver_fifo" if getattr(sim, "net_delay_fn", None) is not None
                             else "fixed_delay_no_contention")
        # Read the actual resources, without adding events or instrumenting their
        # requests/releases. Fixed-delay transfers bypass these resources.
        self.receive_resources = (
            ([port._nvlink for port in sim.ports], [port._rdma for port in sim.ports])
            if self.receive_mode == "receiver_fifo" else None)
        self.windows = {label: _Bucket(start, end, sim.num_gpus, True, self.receive_mode)
                        for label, start, end in self.window_specs}
        self.bins = {}
        self.last_time = 0.0
        self.previous_schedule = [None] * sim.num_gpus
        self.requests = {}
        self.unfinished_order = deque()
        initial = self._state(sim)
        self._bin(0).start_state = initial
        for bucket in self.windows.values():
            if bucket.start == 0:
                bucket.start_state = dict(initial)

    @staticmethod
    def _state(sim):
        return {
            "active_requests": len(sim.requests), "waiting_requests": len(sim.waiting),
            "outstanding_requests": sim.arrived_requests - sim.completed_requests,
            "ready_attention": sim.ready_attention, "ready_expert": sim.ready_expert,
            "incomplete_joins": len(sim.joins), "network_branches": sim.inflight_branches,
            "remaining_output_work": sim.arrived_requests * sim.tokens_per_request - sim.completed_tokens,
        }

    def _bin(self, index):
        if index not in self.bins:
            self.bins[index] = _Bucket(index * self.interval, (index + 1) * self.interval,
                                       self.num_gpus, receive_mode=self.receive_mode)
        return self.bins[index]

    def _ages(self, sim, now):
        while self.unfinished_order and self.requests[self.unfinished_order[0]]["completion_ms"] is not None:
            self.unfinished_order.popleft()
        oldest = (now - self.requests[self.unfinished_order[0]]["arrival_ms"]
                  if self.unfinished_order else 0.0)
        waiting = now - sim.waiting[0][1] / self.ticks_per_ms if sim.waiting else 0.0
        return {"oldest_unfinished_request": oldest, "oldest_waiting_request": waiting}

    def advance(self, sim, now_ticks):
        """Called before env.step(), while sim still holds the preceding state."""
        now = float(now_ticks) / self.ticks_per_ms
        if now < self.last_time:
            raise ValueError("observer time must be monotonic")
        if now == self.last_time:
            return
        state = self._state(sim)
        busy = [int(worker.current_compute is not None) for worker in sim.workers]
        stages = [worker.current_compute[0] if worker.current_compute is not None else None
                  for worker in sim.workers]
        active_cap_full = self.active_cap is not None and len(sim.requests) >= self.active_cap
        receive = (tuple([(resource.count, len(resource.queue)) for resource in channel]
                         for channel in self.receive_resources)
                   if self.receive_resources is not None else None)
        for bucket in self.windows.values():
            if bucket.start_state is None and self.last_time <= bucket.start <= now:
                bucket.start_state = dict(state)
            if bucket.end_state is None and self.last_time <= bucket.end <= now:
                bucket.end_state = dict(state)
                bucket.age_at_end = self._ages(sim, bucket.end)
            bucket.integrate(self.last_time, now, state, busy, stages, active_cap_full, receive)
        cursor = self.last_time
        while cursor < now:
            index = int(math.floor(cursor / self.interval))
            bucket = self._bin(index)
            if bucket.end <= cursor:  # protect a rounded quotient at an exact edge
                index += 1
                bucket = self._bin(index)
            if bucket.start_state is None:
                bucket.start_state = dict(state)
            end = min(now, bucket.end)
            if end <= cursor:
                raise ValueError("sample interval is below clock precision")
            bucket.integrate(cursor, end, state, busy, stages, active_cap_full, receive)
            if end == bucket.end:
                bucket.end_state = dict(state)
                bucket.age_at_end = self._ages(sim, end)
            cursor = end
        # Initialize a new bin's left limit before any event at its left edge.
        at_boundary = self._bin(int(math.floor(now / self.interval)))
        if at_boundary.start_state is None:
            at_boundary.start_state = dict(state)
        self.last_time = now

    def _windows_at(self, now):
        return [bucket for bucket in self.windows.values() if bucket.start <= now < bucket.end]

    def _event(self, now, event):
        self._bin(int(math.floor(now / self.interval))).counts[event] += 1
        for bucket in self._windows_at(now):
            bucket.counts[event] += 1

    def metric(self, now_ticks, name, values_ticks):
        buckets = self._windows_at(float(now_ticks) / self.ticks_per_ms)
        if not buckets:
            return
        values = [float(value) / self.ticks_per_ms for value in values_ticks]
        for bucket in buckets:
            bucket.metrics[name].add(values)

    def arrived(self, sim, rid):
        now = float(sim.env.now) / self.ticks_per_ms
        mapping = sim.routing_request_indices
        ordinal = mapping[rid] if mapping is not None else rid % sim.router.num_profiled_requests
        self.requests[rid] = {"request_id": rid, "routing_request_index": ordinal,
                              "arrival_ms": now, "admission_ms": None, "home_gpu": None,
                              "first_token_completion_ms": None, "last_token_completion_ms": None,
                              "completion_ms": None, "completed_tokens": 0,
                              "admission_wait_ms": None, "request_latency_ms": None}
        self.unfinished_order.append(rid)
        self._event(now, "arrived_requests")

    def admitted(self, sim, rid, home, arrival_ticks):
        now = float(sim.env.now) / self.ticks_per_ms
        self.requests[rid].update(admission_ms=now, home_gpu=home,
                                  admission_wait_ms=now - arrival_ticks / self.ticks_per_ms)
        self._event(now, "admitted_requests")
        self.metric(sim.env.now, "admission_wait", [sim.env.now - arrival_ticks])

    def token_completed(self, sim, token):
        now = float(sim.env.now) / self.ticks_per_ms
        record = self.requests[token.request_id]
        previous = record["last_token_completion_ms"]
        if record["first_token_completion_ms"] is None:
            record["first_token_completion_ms"] = now
        record["last_token_completion_ms"] = now
        record["completed_tokens"] += 1
        self._event(now, "completed_tokens")
        self.metric(sim.env.now, "token_latency", [sim.env.now - token.birth_time])
        if previous is not None:
            self.metric(sim.env.now, "itl", [(now - previous) * self.ticks_per_ms])

    def request_completed(self, sim, rid):
        now = float(sim.env.now) / self.ticks_per_ms
        record = self.requests[rid]
        record.update(completion_ms=now, request_latency_ms=now - record["arrival_ms"])
        self._event(now, "completed_requests")
        self.metric(sim.env.now, "request_latency", [record["request_latency_ms"] * self.ticks_per_ms])

    def scheduled(self, trace):
        now, gpu = float(trace["start"]) / self.ticks_per_ms, trace["gpu"]
        stage, layer = trace["stage"], trace["layer"]
        # A layer-mode expert batch has no expert ID; expert-wise batches
        # change queue even when the stage and layer remain unchanged.
        queue = (stage, layer, trace.get("expert_id") if stage == "expert" else None)
        previous = self.previous_schedule[gpu]
        for bucket in self._windows_at(now):
            bucket.schedules[stage] += 1
            hist = "expert_group" if stage == "expert" else stage
            bucket.histograms[hist][trace["batch_size"]] += 1
            if stage == "expert":
                bucket.histograms["expert_individual"].update(count for count in trace["expert_counts"] if count)
            if previous is not None:
                bucket.queue_switches[gpu] += int(previous != queue)
                bucket.stage_switches[gpu] += int(previous[0] != stage)
                bucket.layer_switches[gpu] += int(previous[1] != layer)
        self.previous_schedule[gpu] = queue

    def finish(self, sim, *, fully_drained):
        if self._finished is not None:
            return self._finished
        simulation_end = float(sim.env.now) / self.ticks_per_ms
        observation_end = (max(simulation_end, max(bucket.end for bucket in self.windows.values()))
                           if fully_drained else simulation_end)
        self.advance(sim, observation_end * self.ticks_per_ms)
        state = self._state(sim)
        timeline = []
        for _, bucket in sorted(self.bins.items()):
            if bucket.duration <= 0 and not any(bucket.counts.values()):
                continue
            final = bucket.start <= observation_end < bucket.end
            if final:
                bucket.age_at_end = self._ages(sim, observation_end)
            timeline.append(bucket.summary(observation_end, state, right_closed=final))
        for bucket in self.windows.values():
            if bucket.start <= observation_end < bucket.end:
                bucket.age_at_end = self._ages(sim, observation_end)
        if self.request_log_path is not None:
            self.request_log_path.parent.mkdir(parents=True, exist_ok=True)
            fields = ("request_id", "routing_request_index", "arrival_ms", "admission_ms", "home_gpu",
                      "first_token_completion_ms", "last_token_completion_ms", "completion_ms", "completed_tokens",
                      "admission_wait_ms", "request_latency_ms", "status")
            with self.request_log_path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for record in self.requests.values():
                    status = ("completed" if record["completion_ms"] is not None else
                              "active" if record["admission_ms"] is not None else "waiting")
                    writer.writerow(dict(record, status=status))
        self._finished = {
            "schema_version": 1, "time_unit": "ms", "sample_interval_ms": self.interval,
            "active_request_cap": self.active_cap, "receive_resource_mode": self.receive_mode,
            "simulation_end_ms": simulation_end, "observation_end_ms": observation_end,
            "fully_drained": bool(fully_drained), "extended_idle_observation": observation_end > simulation_end,
            "windows": {label: bucket.summary(observation_end, state,
                        right_closed=bucket.start <= observation_end < bucket.end)
                        for label, bucket in self.windows.items()},
            "timeline": timeline, "request_log_path": str(self.request_log_path) if self.request_log_path else None,
            "request_log_count": len(self.requests),
            "definitions": {
                "windows": "complete windows count events in [start_ms,end_ms) with left-limit boundary states; partial observation-stop windows include events at stop and post-stop state_at_end, flagged right_closed_at_observation_stop (including zero-duration stop at window start); no unobserved padding except known fully drained idle",
                "timeline": "time-weighted bins; last partial/zero-duration bin includes events at observation stop and uses post-stop state_at_end, flagged right_closed_at_observation_stop",
                "remaining_output_work": "arrived_requests * tokens_per_request - completed_tokens; includes waiting requests",
                "ready_attention": "all attention-ready tokens, including terminal finalize; executing work excluded",
                "age_samples": "left-limit ages at bin/window end; observation-stop partial uses final state; zero means no outstanding item",
                "metrics": "waits attributed to service/admission start; ITL/token/request latencies to completion time; window-local quantiles exact when available, branch ready waits keep moments only",
                "busy": "logical compute service occupancy, not measured GPU kernel utilization",
                "gpu_stage_busy": "time-integral of the executing attention/expert/finalize stage per GPU; clipped at window/bin/observation boundaries; stages sum to total compute busy",
                "active_cap_full": "time with active request count at the global admission cap, divided by observed duration; waiting requests excluded",
                "receive_resources": "per-GPU receiver FIFO resource occupancy and queued message count, split into intra/inter host; independent channels may overlap compute; includes modeled latency plus payload service, not hardware link utilization; same-GPU and zero-delay transfers bypass resources; all receive fields are null for fixed_delay_no_contention",
                "receive_queue_message_ms": "time-integral of waiting transfer messages, excluding the message in service; not expert copies or bytes",
                "queue_switches": "successive scheduled (stage,layer,expert_id) differ; expert_id is null for grouped/attention queues; includes predecessor before the window",
            },
        }
        return self._finished


__all__ = ["SteadyMetricsObserver"]
