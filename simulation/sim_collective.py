"""Barrier EP baselines on the asynchronous simulator's clocks and costs.

One (sync) or two (TBO) fixed microbatches advance within each decode iteration.
A global compute-phase resource serializes their A/E phases; GPUs inside a
phase run in parallel. Dispatch/return use the same receive resources as async
and may overlap another microbatch's compute. This is a logical baseline, not a
model of a particular NCCL collective implementation.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import gzip
import heapq
import json
import math
from pathlib import Path

import simpy

if __package__:
    from .sim_async import AsyncMoESimulator, GPUWorker
else:
    from sim_async import AsyncMoESimulator, GPUWorker


class CollectiveGPUWorker(GPUWorker):
    """A single GPU executing controller commands, never an argmax scheduler."""

    def __init__(self, sim, rank):
        self.sim, self.env, self.rank = sim, sim.env, rank
        self.queue_trace, self.observer = None, sim.observer
        # Shared enqueue/service helpers also support optional async tracing.
        self.token_trace = None
        self.attention = [deque() for _ in range(sim.num_layers + 1)]
        self.experts = [[deque() for _ in range(sim.experts_per_gpu)] for _ in range(sim.num_layers)]
        self._ready_expert_queues = set()
        self.expert_scheduling = "layer"
        self.wakeup = self.env.event()
        self.busy_time, self.current_compute = 0.0, None
        self.busy_by_stage = Counter()
        self.overhead_batch_counts = Counter()
        self.overhead_scheduled_ticks = defaultdict(Counter)
        self.overhead_workload_counts = defaultdict(Counter)
        self.attention_batches, self.finalize_batches = Counter(), Counter()
        self.expert_batches, self.expert_individual_batches = Counter(), Counter()
        self.commands = simpy.Store(self.env)
        self.process = self.env.process(self.run())

    @staticmethod
    def _take_matching(queue, tids, *, branch=False):
        selected, retained = deque(), deque()
        for item in queue:
            token = item[0].token if branch else item[0]
            (selected if token.tid in tids else retained).append(item)
        queue.clear()
        queue.extend(retained)
        return selected

    def run(self):
        while True:
            command = yield self.commands.get()
            stage, layer, tids = command["stage"], command["layer"], command["tids"]
            # The phase owns these items, but they remain counted as waiting
            # until their actual capped compute dispatch below.
            if stage == "expert":
                eligible = [self._take_matching(q, tids, branch=True) for q in self.experts[layer]]
                remaining = sum(map(len, eligible))
            else:
                eligible = self._take_matching(self.attention[layer], tids)
                remaining = len(eligible)
            output, intervals = [], []
            while remaining:
                started = float(self.env.now)
                # Snapshot includes the command's not-yet-dispatched items and
                # other microbatches. It is occupancy, not scheduling eligibility.
                loads, estimate = None, None
                if stage == "expert":
                    items, loads = [], [0] * self.sim.experts_per_gpu
                    heads = [(q[0][1], index) for index, q in enumerate(eligible) if q]
                    heapq.heapify(heads)
                    while heads and len(items) < self.sim.expert_batch_cap:
                        _, index = heapq.heappop(heads)
                        items.append(eligible[index].popleft())
                        loads[index] += 1
                        if eligible[index]:
                            heapq.heappush(heads, (eligible[index][0][1], index))
                    duration = self.sim.expert_cost(loads)
                    self.expert_batches[len(items)] += 1
                    self.expert_individual_batches.update(n for n in loads if n)
                    self.sim.ready_expert -= len(items)
                    column = self.sim.num_layers + 1 + layer
                    metric = "expert_ready_wait"
                else:
                    items = [eligible.popleft() for _ in range(min(remaining, self.sim.attn_batch_cap))]
                    if stage == "finalize":
                        duration = len(items) * self.sim.finalize_t
                        self.finalize_batches[len(items)] += 1
                        metric = "finalize_ready_wait"
                    else:
                        estimate = self.sim.attention_cost.evaluate(
                            [self.sim.initial_context + token.token_index for token, _ in items])
                        duration = estimate.total_ticks
                        self.attention_batches[len(items)] += 1
                        metric = "attention_ready_wait"
                    self.sim.ready_attention -= len(items)
                    column = layer
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError("compute cost must be finite and nonnegative")
                # The controller names a fixed phase; it does not run the
                # async layer-selection scan. Buffer and row work is shared.
                service = self.batch_service(stage, duration, len(items),
                                             items=items, candidate_queues=1)
                duration = service.total_ticks
                snapshot = None
                if self.sim.should_capture_schedule_details(started, started + duration):
                    a, e, _ = self.snapshot()
                    snapshot = a + e
                    snapshot[column] += remaining
                if self.queue_trace is not None:
                    self.queue_trace.change(started, self.rank, column, -len(items))
                for item, ready_at in items:
                    token = item.token if stage == "expert" else item
                    wait = started - ready_at
                    self.sim.stats[metric].add(wait)
                    setattr(token, metric, getattr(token, metric) + wait)
                if self.observer is not None:
                    self.observer.metric(started, metric, (started - at for _, at in items))
                trace = dict(gpu=self.rank, stage=stage, layer=layer,
                             start=started, end=started + duration,
                             start_ms=started / self.sim.ticks_per_ms,
                             end_ms=(started + duration) / self.sim.ticks_per_ms,
                             batch_size=len(items), eligible_queue_size=remaining,
                             expert_counts=loads, expert_scheduling="layer",
                             local_expert_id=None, expert_id=None,
                             selected_queue_index=None, scores=[],
                             iteration=command["iteration"], microbatch=command["microbatch"],
                             phase=command["phase"])
                self.add_overhead_trace(trace, service)
                if estimate is not None:
                    trace.update(attention_context_mean=estimate.context_mean,
                                 attention_components_ticks=estimate.components_ticks)
                if snapshot is not None:
                    trace["queue_snapshot"] = snapshot
                self.sim.capture_schedule(trace)
                gpu_start = started + service.cpu_exposed_ticks
                interval = dict(gpu=self.rank, start=gpu_start, end=started + duration,
                                start_ms=gpu_start / self.sim.ticks_per_ms, end_ms=trace["end_ms"],
                                batch_size=len(items))
                if self.sim.overhead_model.enabled:
                    interval.update(service_start=started, service_start_ms=trace["start_ms"],
                                    compute_start=trace["compute_start"],
                                    batch_overhead=service.as_dict())
                intervals.append(interval)
                yield from self.execute_service(stage, service)
                if self.sim.track_fragments:
                    self.sim.mark_batch_output(stage, items)
                output.extend(item for item, _ in items)
                remaining -= len(items)
            command["done"].succeed((output, intervals))


class CollectiveMoESimulator(AsyncMoESimulator):
    worker_type = CollectiveGPUWorker

    def __init__(self, **kwargs):
        prefix = kwargs["settings"].get("queue_trace_path")
        if prefix is not None and Path(str(Path(prefix).expanduser().resolve()) + ".phases.jsonl.gz").exists():
            raise FileExistsError(f"collective phase trace already exists for prefix: {prefix}")
        super().__init__(**kwargs)
        self.compute_phase = simpy.Resource(self.env, capacity=1)
        self._arrival_wakeup = self.env.event()
        self._transfer_collector = None
        self.collective_phase_trace = []
        self.collective_phase_count = 0
        self._phase_stream = None
        self.phase_trace_path = None
        if self.queue_trace is not None:
            self.queue_trace.configuration.update(
                execution_mode=self.execution_mode,
                queue_definition="arrived, waiting work, including iteration/microbatch barrier-gated work; excludes executing work, network transfers, and incomplete joins")
            self.phase_trace_path = str(Path(self.queue_trace_path).expanduser().resolve()) + ".phases.jsonl.gz"
            self.queue_trace.paths["phases"] = self.phase_trace_path
            try:
                self._phase_stream = gzip.open(self.phase_trace_path, "xt", encoding="utf-8")
            except Exception:
                self.queue_trace.close()
                raise
        self.env.process(self.control())

    def admit(self):
        # The source and completed requests only wake the controller. Actual
        # admission happens at a whole-iteration boundary, never inside a phase.
        if not self._arrival_wakeup.triggered:
            self._arrival_wakeup.succeed()

    def _arrivals_due_now(self):
        if self.arrived_requests == self.total_requests:
            return False
        if self.arrival_times_ms is not None:
            return self.arrival_times_ms[self.arrived_requests] * self.ticks_per_ms <= self.env.now
        return self.arrival_rate == 0

    def control(self):
        iteration = 0
        while True:
            # Include simultaneous arrivals before taking a fixed iteration
            # membership. This does not advance simulation time.
            while self._arrivals_due_now():
                yield self.env.timeout(0)
            super().admit()
            tokens_by_home = [[token for token, _ in worker.attention[0]] for worker in self.workers]
            if not any(tokens_by_home):
                self._arrival_wakeup = self.env.event()
                yield self._arrival_wakeup
                continue
            count = 1 if self.execution_mode == "sync" else 2
            batches = [[] for _ in range(count)]
            for tokens in tokens_by_home:
                for index, token in enumerate(tokens):
                    batches[index % count].append(token)
            processes = [self.env.process(self.microbatch(iteration, mb, tokens))
                         for mb, tokens in enumerate(batches) if tokens]
            yield self.env.all_of(processes)
            iteration += 1

    def _record_phase(self, record):
        record.update(start_ms=record["start"] / self.ticks_per_ms,
                      end_ms=record["end"] / self.ticks_per_ms,
                      time_unit="ticks", ticks_per_millisecond=self.ticks_per_ms)
        self.collective_phase_count += 1
        if len(self.collective_phase_trace) < self.trace_limit:
            self.collective_phase_trace.append(record)
        if self._phase_stream is not None and self.queue_trace.observes_schedule(record["start"], record["end"]):
            self._phase_stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    def _phase_record(self, iteration, mb, layer, phase, start, tokens, **extra):
        return dict(iteration=iteration, microbatch=mb, layer=layer, phase=phase,
                    start=start, end=float(self.env.now), token_count=len(tokens),
                    token_counts_by_home=[sum(t.home_gpu == gpu for t in tokens) for gpu in range(self.num_gpus)],
                    **extra)

    def compute(self, iteration, mb, layer, stage, tokens):
        phase = stage + "_compute"
        with self.compute_phase.request() as lock:
            yield lock
            start = float(self.env.now)
            tids = {token.tid for token in tokens}
            commands = []
            for worker in self.workers:
                done = self.env.event()
                worker.commands.put(dict(stage=stage, layer=layer, tids=tids,
                                         iteration=iteration, microbatch=mb, phase=phase, done=done))
                commands.append(done)
            results = yield self.env.all_of(commands)
            by_gpu = [results[event][0] for event in commands]
            intervals = [interval for event in commands for interval in results[event][1]]
            self._record_phase(self._phase_record(iteration, mb, layer, phase, start, tokens,
                                                 resource="global_compute_phase", gpu_intervals=intervals))
        return by_gpu

    def send(self, src, dst, branches, returning):
        self.inflight_branches += len(branches)
        self.peak_inflight_branches = max(self.peak_inflight_branches, self.inflight_branches)
        if src != dst:
            self.remote_transfers += 1
            self.remote_copies += len(branches)
        transfer = self.env.process(self._deliver(src, dst, branches, returning))
        self._transfer_collector.append(transfer)
        return transfer

    def communicate(self, iteration, mb, layer, tokens, *, computed=None):
        start = float(self.env.now)
        self._transfer_collector = []
        if computed is None:
            for gpu in range(self.num_gpus):
                local = [token for token in tokens if token.home_gpu == gpu]
                if local:
                    self.dispatch(gpu, layer, local)
            phase = "dispatch"
        else:
            for src, branches in enumerate(computed):
                destinations = defaultdict(list)
                for branch in branches:
                    destinations[branch.token.home_gpu].append(branch)
                for dst, outgoing in destinations.items():
                    self.send(src, dst, outgoing, returning=True)
            phase = "combine"
        transfers, self._transfer_collector = self._transfer_collector, None
        yield self.env.all_of(transfers)
        self._record_phase(self._phase_record(iteration, mb, layer, phase, start, tokens,
                                             resource="receive_ports", transfer_count=len(transfers)))

    def microbatch(self, iteration, mb, tokens):
        for layer in range(self.num_layers):
            yield self.env.process(self.compute(iteration, mb, layer, "attention", tokens))
            yield self.env.process(self.communicate(iteration, mb, layer, tokens))
            computed = yield self.env.process(self.compute(iteration, mb, layer, "expert", tokens))
            yield self.env.process(self.communicate(iteration, mb, layer, tokens, computed=computed))
        yield self.env.process(self.compute(iteration, mb, self.num_layers, "finalize", tokens))
        for token in tokens:
            self.complete(token)

    def run(self):
        try:
            result = super().run()
            result.update(execution_mode=self.execution_mode,
                          collective_phase_trace=self.collective_phase_trace,
                          collective_phase_count=self.collective_phase_count,
                          collective_phase_trace_truncated=self.collective_phase_count > len(self.collective_phase_trace),
                          collective_phase_trace_path=self.phase_trace_path,
                          collective_semantics={
                              "microbatches": 1 if self.execution_mode == "sync" else 2,
                              "split": "alternate within each home GPU",
                              "admission": "whole decode iteration boundaries",
                              "next_token_birth": "previous token completion; compute gated until next iteration",
                              "compute": "one global phase; participating GPUs run concurrently, local capped chunks serially",
                              "communication": "per source/destination microbatch aggregation after global compute barrier; same async receive cost/resources",
                              "phase_order": "attention_compute -> dispatch -> expert_compute -> combine; finalize_compute after last layer",
                              "queue_occupancy": "includes arrived barrier-gated work"})
            return result
        finally:
            if self._phase_stream is not None:
                self._phase_stream.close()
