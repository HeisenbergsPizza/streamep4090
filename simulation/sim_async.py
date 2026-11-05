"""CPU discrete-event model of colocated, unified Attention/Expert scheduling.

Compute costs are explicit inputs, not a GPU performance prediction. All time
values in the API are ticks; one tick defaults to 0.1 ms. See README.md.
"""
from __future__ import annotations

import math
import heapq
import os
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable

import simpy
import torch

if __package__:
    from .util import expert_compute_time_lookup_table_from_profile, build_profile_router
    from .costs import AttentionCostModel, ExpertCostModel
    from .overheads import BatchOverheadModel, BatchOverheadEstimate, BatchWorkload
    from .schedulers import UnifiedScheduler, ExpertWiseScheduler
    from .metrics import Samples, percentile
    from .queue_trace import QueueTraceRecorder
    from .token_trace import TokenTraceRecorder
else:  # direct script execution
    from util import expert_compute_time_lookup_table_from_profile, build_profile_router
    from costs import AttentionCostModel, ExpertCostModel
    from overheads import BatchOverheadModel, BatchOverheadEstimate, BatchWorkload
    from schedulers import UnifiedScheduler, ExpertWiseScheduler
    from metrics import Samples, percentile
    from queue_trace import QueueTraceRecorder
    from token_trace import TokenTraceRecorder

EP_GROUP_SIZE = 16
TOTAL_EXPERT_COUNT = 128
MAX_BATCH_SIZE = 512
NUM_LAYERS = 48
TOTAL_REQUESTS = 8192
TOKENS_PER_REQUEST = 256
TOTAL_TOKENS = TOTAL_REQUESTS * TOKENS_PER_REQUEST
GLOBAL_REQUEST_MAX_BATCH_SIZE = EP_GROUP_SIZE * 256
ARRIVAL_RATE = 50.0  # requests / tick
ATTN_SERVICE_T = 2.0  # fallback time per token, not a measurement
ATTN_DP_GROUP_SIZE = EP_GROUP_SIZE
TICKS_PER_MILLISECOND = 10
NET_T_EXPERT_TO_ATTN = 0.1
NET_T_ATTN_TO_EXPERT = 0.1
GROUP_GEMM_SPEEDUP_FACTOR = 1.0  # retained name; no implicit speedup
RNG_SEED = 42
ROUTING_TOP_K = 8
SCHEDULING_POLICY = "defrag"
EXPERT_COMPUTE_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "expert_costs_profiles", "Qwen3-30B.csv")
PROFILE_ROUTING_PATH = os.path.join(os.path.dirname(__file__), "..", "gating_profiles", "gating_sharegptv3_155.parquet")


@dataclass
class Token:
    tid: int
    birth_time: float
    request_id: int
    token_index: int
    home_gpu: int = 0
    layer_fanout: dict[int, int] = field(default_factory=dict)
    sampled_for_stats: bool = False
    previous_completion_time: float | None = None
    attention_ready_wait: float = 0.0
    expert_ready_wait: float = 0.0  # sum over branches, not token critical-path wait
    finalize_ready_wait: float = 0.0
    join_wait: float = 0.0  # sum of first-to-last return gaps over layers
    reformation_inputs: tuple[tuple[int, int], ...] = ()  # logical (buffer, row), current layer only
    output_batch_id: int | None = None


@dataclass
class Branch:
    token: Token
    expert_id: int
    layer: int
    buffer_id: int | None = None
    buffer_row: int = 0
    output_batch_id: int | None = None


@dataclass
class Join:
    token: Token
    remaining: set[int]
    first_return: float | None = None
    input_rows: list[tuple[int, int]] = field(default_factory=list)


def _validate_workload_overrides(sim):
    arrivals = getattr(sim, "arrival_times_ms", None)
    mapping = getattr(sim, "routing_request_indices", None)
    if arrivals is not None:
        arrivals = tuple(arrivals)
        if len(arrivals) != sim.total_requests:
            raise ValueError("arrival_times_ms length must match total_requests")
        previous = 0.0
        for value in arrivals:
            if isinstance(value, bool) or not math.isfinite(value) or value < previous:
                raise ValueError("arrival_times_ms must be finite, nonnegative and monotonic")
            if not math.isfinite(value * sim.ticks_per_ms):
                raise ValueError("arrival_times_ms exceeds the simulation clock range")
            previous = value
        arrivals = tuple(float(value) for value in arrivals)
    if mapping is not None:
        mapping = tuple(mapping)
        if len(mapping) != sim.total_requests:
            raise ValueError("routing_request_indices length must match total_requests")
        if any(isinstance(index, bool) or not isinstance(index, Integral)
               or not 0 <= index < sim.router.num_profiled_requests for index in mapping):
            raise ValueError("routing_request_indices must contain valid profile request ordinals")
    if arrivals is not None or mapping is not None:
        selected = set(mapping) if mapping is not None else range(min(sim.total_requests, sim.router.num_profiled_requests))
        if any(len(sim.router.profiled_token_indices[index]) < sim.tokens_per_request for index in selected):
            raise ValueError("selected routing request has insufficient tokens; explicit workloads cannot wrap token routes")
    sim.arrival_times_ms, sim.routing_request_indices = arrivals, mapping


class IngressPort:
    """Two independent FIFO receive resources; no sender/NVSwitch contention.

    Same-GPU transfers bypass both queues. Network delay includes service time
    only; waiting for a busy receive resource is accounted separately.
    """
    def __init__(self, env, gpu_idx, n_gpu_per_host, wait_stats=None, observer=None):
        self.env = env
        self.gpu_idx = gpu_idx
        self.n_gpu_per_host = n_gpu_per_host
        self._nvlink = simpy.Resource(env, capacity=1)
        self._rdma = simpy.Resource(env, capacity=1)
        self.wait_stats = wait_stats
        self.observer = observer

    def transfer(self, src_gpu, solo_delay):
        if not math.isfinite(solo_delay) or solo_delay < 0:
            raise ValueError("network delay must be finite and nonnegative")
        if src_gpu == self.gpu_idx or solo_delay == 0:
            return self.env.timeout(0)
        intra = src_gpu // self.n_gpu_per_host == self.gpu_idx // self.n_gpu_per_host
        return self.env.process(self._transfer(self._nvlink if intra else self._rdma, solo_delay))

    def _transfer(self, resource, delay):
        start = self.env.now
        with resource.request() as request:
            yield request
            if self.wait_stats is not None:
                self.wait_stats.add(self.env.now - start)
            if self.observer is not None:
                self.observer.metric(self.env.now, "network_receive_wait", [self.env.now - start])
            yield self.env.timeout(delay)


class GPUWorker:
    """One non-preemptive compute resource, with per-stage/per-layer queues."""
    def __init__(self, sim, rank):
        self.sim, self.env, self.rank = sim, sim.env, rank
        self.queue_trace = getattr(sim, "queue_trace", None)
        self.token_trace = getattr(sim, "token_trace", None)
        self.observer = getattr(sim, "observer", None)
        self.attention = [deque() for _ in range(sim.num_layers + 1)]
        self.experts = [[deque() for _ in range(sim.experts_per_gpu)] for _ in range(sim.num_layers)]
        self._ready_expert_queues = set()
        self._expert_queues_initialized = False
        self.expert_scheduling = getattr(sim, "expert_scheduling", "layer")
        scheduler_type = ExpertWiseScheduler if self.expert_scheduling == "expert" else UnifiedScheduler
        scheduler_options = ({"experts_per_layer": sim.experts_per_gpu}
                             if self.expert_scheduling == "expert" else {})
        self.scheduler = scheduler_type(
            sim.num_layers, sim.top_k, sim.policy,
            sim.lookahead_steps, sim.lookback_steps, sim.weight_decay,
            **scheduler_options,
        )
        self.wakeup = self.env.event()
        self.busy_time = 0.0
        self.busy_by_stage = Counter()
        self.overhead_batch_counts = Counter()
        self.overhead_scheduled_ticks = defaultdict(Counter)
        self.overhead_workload_counts = defaultdict(Counter)
        self.current_compute = None
        self.attention_batches = Counter()
        self.finalize_batches = Counter()
        self.expert_batches = Counter()
        self.expert_individual_batches = Counter()
        self.process = self.env.process(self.run())

    def wake(self):
        if not self.wakeup.triggered:
            self.wakeup.succeed()

    def enqueue_attention(self, layer, token):
        self.attention[layer].append((token, self.env.now))
        if self.queue_trace is not None:
            self.queue_trace.change(self.env.now, self.rank, layer, 1)
        self.sim.ready_attention += 1
        self.sim.peak_ready_attention = max(self.sim.peak_ready_attention, self.sim.ready_attention)
        self.wake()

    def enqueue_expert(self, branch):
        local = branch.expert_id % self.sim.experts_per_gpu
        self.experts[branch.layer][local].append((branch, self.env.now))
        self._ready_expert_queues.add((branch.layer, local))
        if self.queue_trace is not None:
            self.queue_trace.change(self.env.now, self.rank, self.sim.num_layers + 1 + branch.layer, 1)
        self.sim.ready_expert += 1
        self.sim.peak_ready_expert = max(self.sim.peak_ready_expert, self.sim.ready_expert)
        if self.token_trace is not None and self.token_trace.pending:
            self.token_trace.expert_ready(branch, self.env.now)
        self.wake()

    def snapshot(self):
        """Layer aggregates for occupancy metrics, independent of scheduling granularity."""
        attn = [len(q) for q in self.attention]
        expert = [sum(len(q) for q in queues) for queues in self.experts]
        oldest = [q[0][1] if q else math.inf for q in self.attention]
        oldest += [min((q[0][1] for q in queues if q), default=math.inf) for queues in self.experts]
        return attn, expert, oldest

    def run(self):
        while True:
            if self.expert_scheduling == "expert":
                # Initialize once to also support directly preloaded test queues.
                # Afterwards enqueue/dequeue maintain this set without scanning
                # every empty expert queue on every GPU dispatch.
                if not self._expert_queues_initialized:
                    self._ready_expert_queues.update(
                        (layer, expert) for layer, queues in enumerate(self.experts)
                        for expert, queue in enumerate(queues) if queue)
                    self._expert_queues_initialized = True
                attn = [len(q) for q in self.attention]
                expert = [(layer, local, len(self.experts[layer][local]))
                          for layer, local in sorted(self._ready_expert_queues)]
                oldest = None
                if self.scheduler.policy == "fifo":
                    oldest = [q[0][1] if q else math.inf for q in self.attention]
                    oldest += [self.experts[layer][local][0][1] for layer, local, _ in expert]
                decision = self.scheduler.select_sparse(attn, expert, oldest, record_details=False)
            else:
                attn = [len(q) for q in self.attention]
                expert = [sum(len(q) for q in queues) for queues in self.experts]
                oldest = self.snapshot()[2] if self.scheduler.policy == "fifo" else None
                decision = self.scheduler.select(attn, expert, oldest)
            if decision is None:
                self.wakeup = self.env.event()
                yield self.wakeup
                continue
            stage, layer = decision[:2]
            local_expert = decision[2] if self.expert_scheduling == "expert" else None
            started = float(self.env.now)
            loads = None
            attention_estimate = None
            if stage == "attention":
                queue = self.attention[layer]
                items = [queue.popleft() for _ in range(min(len(queue), self.sim.attn_batch_cap))]
                tokens = [item[0] for item in items]
                if self.queue_trace is not None:
                    self.queue_trace.change(self.env.now, self.rank, layer, -len(tokens))
                self.sim.ready_attention -= len(tokens)
                for token, ready_at in items:
                    wait = started - ready_at
                    if layer == self.sim.num_layers:
                        self.sim.stats["finalize_ready_wait"].add(wait)
                        token.finalize_ready_wait += wait
                    else:
                        self.sim.stats["attention_ready_wait"].add(wait)
                        token.attention_ready_wait += wait
                if self.observer is not None:
                    name = "finalize_ready_wait" if layer == self.sim.num_layers else "attention_ready_wait"
                    self.observer.metric(self.env.now, name, (started - ready_at for _, ready_at in items))
                contexts = [self.sim.initial_context + token.token_index for token in tokens]
                if layer == self.sim.num_layers:
                    duration = len(tokens) * self.sim.finalize_t
                else:
                    attention_estimate = self.sim.attention_cost.evaluate(contexts)
                    duration = attention_estimate.total_ticks
                batch_size = len(tokens)
                (self.finalize_batches if layer == self.sim.num_layers else self.attention_batches)[batch_size] += 1
            else:
                queues = self.experts[layer]
                items, loads = [], [0] * len(queues)
                if local_expert is not None:
                    queue = queues[local_expert]
                    count = min(len(queue), self.sim.expert_batch_cap)
                    items = [queue.popleft() for _ in range(count)]
                    loads[local_expert] = count
                    if not queue:
                        self._ready_expert_queues.discard((layer, local_expert))
                else:
                    # Merge group queue heads by ready time. Low expert IDs
                    # must not displace older branches when the group cap hits.
                    heads = [(queue[0][1], index) for index, queue in enumerate(queues) if queue]
                    heapq.heapify(heads)
                    while heads and len(items) < self.sim.expert_batch_cap:
                        _, index = heapq.heappop(heads)
                        queue = queues[index]
                        items.append(queue.popleft())
                        loads[index] += 1
                        if queue:
                            heapq.heappush(heads, (queue[0][1], index))
                for take in loads:
                    if take:
                        self.expert_individual_batches[take] += 1
                branches = [item[0] for item in items]
                if self.queue_trace is not None:
                    self.queue_trace.change(self.env.now, self.rank, self.sim.num_layers + 1 + layer, -len(branches))
                self.sim.ready_expert -= len(branches)
                for branch, ready_at in items:
                    wait = started - ready_at
                    self.sim.stats["expert_ready_wait"].add(wait)
                    branch.token.expert_ready_wait += wait
                if self.observer is not None:
                    self.observer.metric(self.env.now, "expert_ready_wait", (started - ready_at for _, ready_at in items))
                duration = self.sim.expert_cost(loads)
                batch_size = len(branches)
                self.expert_batches[batch_size] += 1
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("compute cost must be finite and nonnegative")
            compute_stage = "finalize" if layer == self.sim.num_layers else stage
            service = self.batch_service(compute_stage, duration, batch_size,
                                         items=items, candidate_queues=len(attn) + len(expert))
            duration = service.total_ticks
            trace = {
                "gpu": self.rank, "stage": compute_stage,
                "layer": layer, "start": started, "end": started + duration,
                "batch_size": batch_size, "expert_counts": loads,
                "expert_scheduling": self.expert_scheduling,
                "local_expert_id": local_expert,
                "expert_id": (self.rank * self.sim.experts_per_gpu + local_expert
                              if local_expert is not None else None),
                "selected_queue_index": self.scheduler.last_selected_index,
            }
            self.add_overhead_trace(trace, service)
            if attention_estimate is not None:
                # Planned cost breakdown, not a measured sub-kernel timeline.
                trace["attention_context_mean"] = attention_estimate.context_mean
                trace["attention_components_ticks"] = attention_estimate.components_ticks
            capture_details = getattr(self.sim, "should_capture_schedule_details", None)
            if capture_details is None or capture_details(started, started + duration):
                if self.expert_scheduling == "expert":
                    self.scheduler.materialize_details()
                trace["queue_snapshot"] = self.scheduler.last_snapshot[:]
                trace["scores"] = self.scheduler.last_scores[:]
            self.sim.capture_schedule(trace)
            token_trace = getattr(self, "token_trace", None)
            token_trace_batch_id = (token_trace.batch_selected(trace, items)
                                    if token_trace is not None else None)
            yield from self.execute_service(compute_stage, service, token_trace_batch_id)
            if token_trace is not None:
                token_trace.batch_completed(token_trace_batch_id, self.env.now)
            if getattr(self.sim, "track_fragments", False):
                self.sim.mark_batch_output(compute_stage, items)
            if stage == "attention":
                if layer == self.sim.num_layers:
                    for token in tokens:
                        self.sim.complete(token)
                else:
                    self.sim.dispatch(self.rank, layer, tokens)
            else:
                # Return communication is independent of the compute resource.
                destinations = defaultdict(list)
                for branch in branches:
                    destinations[branch.token.home_gpu].append(branch)
                for dst, outgoing in destinations.items():
                    self.sim.send(self.rank, dst, outgoing, returning=True)

    def batch_workload(self, stage, items, candidate_queues):
        """Count selected source-row spans, without materializing activation tensors.

        Forward payload rows are ordered by expert. Return payloads contain
        the newly produced expert outputs. A capped selection can therefore
        retain one contiguous slice or require gathering several disjoint ones.
        This is logical buffer provenance, not a physical allocator model.
        """
        if stage == "expert":
            refs = [(branch.buffer_id, branch.buffer_row) for branch, _ in items]
        else:
            refs = [ref for token, _ in items for ref in token.reformation_inputs]
        if not refs or any(buffer is None for buffer, _ in refs):
            raise RuntimeError("structural overhead requires source-buffer provenance")
        by_buffer = defaultdict(set)
        for buffer, row in refs:
            by_buffer[buffer].add(row)
        if sum(map(len, by_buffer.values())) != len(refs):
            raise RuntimeError("duplicate source row in selected batch")
        fragments = 0
        for rows in by_buffer.values():
            fragments += sum(row - 1 not in rows for row in rows)
        contiguous = (stage == "expert" and fragments == 1
                      and all(refs[i] == (refs[0][0], refs[0][1] + i)
                              for i in range(len(refs))))
        return BatchWorkload(input_rows=len(refs), fragments=fragments,
                             candidate_queues=candidate_queues,
                             hidden_dim=self.sim.hidden_dim,
                             bytes_per_element=self.sim.bytes_per_element,
                             contiguous=contiguous)

    def batch_service(self, stage, duration, batch_size, *, items=None, candidate_queues=1):
        model = getattr(self.sim, "overhead_model", None)
        if model is None or not model.enabled:
            return BatchOverheadEstimate(duration)
        workload = None
        if model.preset == "structural" and stage != "finalize" and batch_size:
            workload = self.batch_workload(stage, items, candidate_queues)
        estimate = model.evaluate(stage, duration, batch_size=batch_size, workload=workload)
        if stage != "finalize" and batch_size:
            self.overhead_batch_counts[stage] += 1
            for name, value in estimate.as_dict().items():
                self.overhead_scheduled_ticks[stage][name] += value
            if estimate.details is not None:
                features = estimate.details["workload"]
                counts = self.overhead_workload_counts[stage]
                counts["output_rows"] += batch_size
                for key in ("input_rows", "fragments", "candidate_queues",
                            "read_bytes", "write_bytes", "traffic_bytes"):
                    counts[key] += features[key]
                counts["fast_path_batches"] += int(estimate.details["components"]["fast_path"])
                counts["executed_traffic_bytes"] += estimate.details["components"]["executed_traffic_bytes"]
        return estimate

    def overhead_accounting(self):
        result = {"batch_counts": dict(self.overhead_batch_counts),
                  "by_stage_ticks": {stage: dict(values) for stage, values
                                     in self.overhead_scheduled_ticks.items()}}
        if self.sim.overhead_model.preset == "structural":
            result["by_stage_workload"] = {stage: dict(values) for stage, values
                                          in self.overhead_workload_counts.items()}
        return result

    def add_overhead_trace(self, trace, service):
        model = getattr(self.sim, "overhead_model", None)
        if model is not None and model.enabled:
            # start/end retain the complete reserved batch service interval.
            trace["batch_overhead"] = service.as_dict()
            trace["gpu_start"] = trace["start"] + service.cpu_exposed_ticks
            trace["compute_start"] = trace["gpu_start"] + service.reformation_exposed_ticks
            trace["compute_end"] = trace["end"]
            if service.details is not None:
                trace["batch_workload"] = service.details["workload"]
                trace["batch_overhead_components"] = service.details["components"]

    def execute_service(self, stage, service, token_trace_batch_id=None):
        # A fixed exposure assumption, not a simulated host thread. The batch
        # has already been reserved; arrivals during this gap cannot join it.
        if service.cpu_exposed_ticks > 0:
            yield self.env.timeout(service.cpu_exposed_ticks)
        self.current_compute = (stage, float(self.env.now), service.gpu_ticks)
        token_trace = getattr(self, "token_trace", None)
        if token_trace is not None:
            token_trace.batch_gpu_started(token_trace_batch_id, self.env.now)
        # Preserve the original single event, including zero compute, when off.
        yield self.env.timeout(service.gpu_ticks)
        self.busy_time += service.gpu_ticks
        self.busy_by_stage[stage] += service.gpu_ticks
        self.current_compute = None

    def observed_busy(self):
        busy = self.busy_time
        stages = dict(self.busy_by_stage)
        if self.current_compute is not None:
            stage, start, duration = self.current_compute
            elapsed = max(0.0, min(duration, self.env.now - start))
            busy += elapsed
            stages[stage] = stages.get(stage, 0.0) + elapsed
        return busy, stages


class AsyncMoESimulator:
    worker_type = GPUWorker

    def __init__(self, *, settings, router, expert_cost, attention_cost):
        for name, value in settings.items():
            setattr(self, name, value)
        self.env = simpy.Environment()
        self.router, self.expert_cost, self.attention_cost = router, expert_cost, attention_cost
        self.overhead_model = BatchOverheadModel(
            ticks_per_millisecond=self.ticks_per_ms,
            **(getattr(self, "overhead_config", None) or {}))
        self.hidden_dim = getattr(self, "hidden_dim", 2048)
        self.bytes_per_element = getattr(self, "bytes_per_element", 2)
        self.track_fragments = self.overhead_model.preset == "structural"
        self.next_buffer_id = 0
        self.next_output_batch_id = 0
        self.expert_scheduling = getattr(self, "expert_scheduling", "layer")
        if self.expert_scheduling not in ("layer", "expert"):
            raise ValueError("expert_scheduling must be 'layer' or 'expert'")
        _validate_workload_overrides(self)
        self.observer = getattr(self, "observer", None)
        self.experts_per_gpu = self.num_experts // self.num_gpus
        self.queue_trace = None
        self.token_trace = None
        self.stats = {name: Samples(self.metric_sample_cap, self.seed + i) for i, name in enumerate((
            "attention_ready_wait", "expert_ready_wait", "finalize_ready_wait", "join_wait", "network_receive_wait",
            "admission_wait", "token_latency", "itl", "request_latency",
        ))}
        self.ports = [IngressPort(self.env, rank, self.gpus_per_host, self.stats["network_receive_wait"], self.observer)
                      for rank in range(self.num_gpus)]
        self.workers = [self.worker_type(self, rank) for rank in range(self.num_gpus)]
        self.requests = {}  # rid -> (arrival, home GPU, next token index)
        self.waiting = deque()
        self.joins = {}
        self.next_tid = 0
        self.next_home = 0
        self.completed_requests = 0
        self.completed_tokens = 0
        self.completion_times = {}
        self.schedule_trace = []
        self.schedule_count = 0
        self.last_debug = None
        self.saturation_time = None
        self.saturation_tokens = None
        self.saturation_step = None
        self.steps = 0
        self.arrived_requests = 0
        self.ready_attention = self.ready_expert = 0
        self.peak_ready_attention = self.peak_ready_expert = 0
        self.peak_active = self.peak_waiting = self.peak_joins = 0
        self.inflight_branches = self.peak_inflight_branches = 0
        self.remote_transfers = self.remote_copies = 0
        self.env.process(self.source())
        # Open artifact files only after simulator construction can no longer fail.
        # No SimPy process runs until run(), so all queue hooks are still at zero.
        if getattr(self, "queue_trace_path", None) is not None:
            self.queue_trace = QueueTraceRecorder(
                self.queue_trace_path, num_gpus=self.num_gpus, num_layers=self.num_layers,
                ticks_per_millisecond=self.ticks_per_ms,
                interval_ms=self.queue_trace_interval_ms, start_ms=self.queue_trace_start_ms,
                end_ms=self.queue_trace_end_ms, policy=self.policy, top_k=self.top_k,
                configuration={key: getattr(self, key) for key in (
                    "num_gpus", "num_experts", "num_layers", "expert_batch_cap", "attn_batch_cap", "expert_scheduling",
                    "total_requests", "tokens_per_request", "arrival_rate", "active_cap",
                    "initial_context", "finalize_t", "dispatch_t", "return_t", "gpus_per_host",
                    "seed", "lookahead_steps", "lookback_steps", "weight_decay", "stop_after")})
            for worker in self.workers:
                worker.queue_trace = self.queue_trace
            self.queue_trace.configuration["batch_overhead_model"] = self.overhead_model.describe()
        try:
            if getattr(self, "token_trace_path", None) is not None:
                if getattr(self, "execution_mode", "async") != "async":
                    raise ValueError("token tracing supports execution_mode='async' only")
                self.token_trace = TokenTraceRecorder(
                    self.token_trace_path, num_layers=self.num_layers,
                    ticks_per_millisecond=self.ticks_per_ms,
                    start_ms=getattr(self, "token_trace_start_ms", 0.0),
                    end_ms=getattr(self, "token_trace_end_ms", None),
                    configuration={key: getattr(self, key) for key in (
                        "num_gpus", "num_experts", "num_layers", "top_k", "expert_batch_cap",
                        "attn_batch_cap", "expert_scheduling", "policy", "active_cap", "seed")})
                self.token_trace.configuration["batch_overhead_model"] = self.overhead_model.describe()
                for worker in self.workers:
                    worker.token_trace = self.token_trace
            if self.observer is not None:
                self.observer.bind(self)
        except Exception:
            if self.queue_trace is not None:
                self.queue_trace.close()
            if self.token_trace is not None:
                self.token_trace.close(self.env.now, reason="simulator_initialization_failed")
            raise

    def source(self):
        rng = random.Random(self.seed)
        for rid in range(self.total_requests):
            if self.arrival_times_ms is not None:
                delay = self.arrival_times_ms[rid] * self.ticks_per_ms - self.env.now
                if rid or delay > 0:
                    yield self.env.timeout(delay)
            elif rid:
                yield self.env.timeout(rng.expovariate(self.arrival_rate) if self.arrival_rate > 0 else 0)
            self.arrived_requests += 1
            self.waiting.append((rid, float(self.env.now)))
            if self.observer is not None:
                self.observer.arrived(self, rid)
            self.admit()
            self.peak_waiting = max(self.peak_waiting, len(self.waiting))

    def admit(self):
        while self.waiting and len(self.requests) < self.active_cap:
            rid, arrival = self.waiting.popleft()
            home = self.next_home % self.num_gpus
            self.next_home += 1
            self.requests[rid] = (arrival, home, 0)
            if self.observer is not None:
                self.observer.admitted(self, rid, home, arrival)
            self.stats["admission_wait"].add(self.env.now - arrival)
            self.peak_active = max(self.peak_active, len(self.requests))
            if self.saturation_time is None and len(self.requests) >= self.active_cap:
                self.saturation_time = float(self.env.now)
                self.saturation_tokens = self.completed_tokens
                self.saturation_step = self.steps
            self.spawn_token(rid)

    def spawn_token(self, rid):
        arrival, home, index = self.requests[rid]
        token = Token(self.next_tid, float(self.env.now), rid, index, home)
        if self.track_fragments:
            # Independent decode admissions have independent input rows. This
            # assumption deliberately does not reconstruct tokenizer batching.
            token.reformation_inputs = ((-token.tid - 1, 0),)
        if index > 0:
            token.previous_completion_time = float(self.env.now)
        self.next_tid += 1
        self.requests[rid] = (arrival, home, index + 1)
        if self.begin_cb is not None:
            self.begin_cb(token)
        self.workers[home].enqueue_attention(0, token)

    def dispatch(self, src, layer, tokens):
        _, ids = self.router.route(
            request_ids=([t.request_id for t in tokens] if self.routing_request_indices is None else
                         [self.routing_request_indices[t.request_id] for t in tokens]),
            token_indices=[t.token_index for t in tokens], layer_id=layer,
            top_k=self.top_k, device=torch.device("cpu"), dtype=torch.float32,
        )
        destinations = defaultdict(list)
        for token, expert_ids in zip(tokens, ids.tolist()):
            selected = [eid for eid in expert_ids if eid >= 0]
            if not selected or len(set(selected)) != len(selected):
                raise ValueError("routing must select nonempty, distinct expert IDs")
            token.layer_fanout[layer] = len(selected)
            self.joins[(token.tid, layer)] = Join(token, set(selected))
            if self.token_trace is not None and self.token_trace.pending:
                self.token_trace.routed(token, layer, selected, self.experts_per_gpu, self.env.now)
            for eid in selected:
                destinations[eid // self.experts_per_gpu].append(Branch(token, eid, layer))
        self.peak_joins = max(self.peak_joins, len(self.joins))
        for dst, branches in destinations.items():
            self.send(src, dst, branches, returning=False)

    def send(self, src, dst, branches, returning):
        self.inflight_branches += len(branches)
        self.peak_inflight_branches = max(self.peak_inflight_branches, self.inflight_branches)
        if src != dst:
            self.remote_transfers += 1
            self.remote_copies += len(branches)
        self.env.process(self._deliver(src, dst, branches, returning))

    def mark_batch_output(self, stage, items):
        if self.track_fragments and stage != "finalize":
            batch_id = self.next_output_batch_id
            self.next_output_batch_id += 1
            for item, _ in items:
                item.output_batch_id = batch_id

    def _deliver(self, src, dst, branches, returning):
        if self.track_fragments:
            # A collective may coalesce several capped compute batches into
            # one transfer. Coalescing messages must not silently turn their
            # distinct source tensors into one contiguous buffer for free.
            producers = defaultdict(list)
            for branch in branches:
                producer = branch if returning else branch.token
                if producer.output_batch_id is None:
                    raise RuntimeError("structural transfer requires producing-batch provenance")
                producers[producer.output_batch_id].append(branch)
            for group in producers.values():
                buffer_id = self.next_buffer_id
                self.next_buffer_id += 1
                rows = group if returning else sorted(group, key=lambda branch: branch.expert_id)
                for row, branch in enumerate(rows):
                    branch.buffer_id, branch.buffer_row = buffer_id, row
        if src == dst:
            delay = 0.0
        elif self.net_delay_fn is not None:
            delay = float(self.net_delay_fn(src, dst, len(branches)))
        else:
            delay = self.return_t if returning else self.dispatch_t
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("network delay must be finite and nonnegative")
        if self.net_delay_fn is not None:
            yield self.ports[dst].transfer(src, delay)
        else:
            yield self.env.timeout(delay)
        self.inflight_branches -= len(branches)
        # Each destination is released independently; there is no fanout barrier.
        for branch in branches:
            if returning:
                self.join(branch)
            else:
                self.workers[dst].enqueue_expert(branch)

    def join(self, branch):
        key = (branch.token.tid, branch.layer)
        state = self.joins.get(key)
        if state is None or branch.expert_id not in state.remaining:
            raise RuntimeError(f"duplicate or unexpected expert completion: {key}, {branch.expert_id}")
        if self.token_trace is not None and self.token_trace.pending:
            self.token_trace.returned(branch, self.env.now)
        state.remaining.remove(branch.expert_id)
        if self.track_fragments:
            state.input_rows.append((branch.buffer_id, branch.buffer_row))
        if state.first_return is None:
            state.first_return = float(self.env.now)
        if not state.remaining:
            wait = self.env.now - state.first_return
            self.stats["join_wait"].add(wait)
            if self.observer is not None:
                self.observer.metric(self.env.now, "join_wait", [wait])
            state.token.join_wait += wait
            del self.joins[key]
            if self.token_trace is not None and self.token_trace.pending:
                self.token_trace.joined(state.token, branch.layer, self.env.now)
            if self.track_fragments:
                state.token.reformation_inputs = tuple(state.input_rows)
            self.workers[state.token.home_gpu].enqueue_attention(branch.layer + 1, state.token)

    def complete(self, token):
        now = float(self.env.now)
        if token.tid in self.completion_times:
            raise RuntimeError("token completed twice")
        self.completion_times[token.tid] = now
        self.completed_tokens += 1
        if self.observer is not None:
            self.observer.token_completed(self, token)
        self.stats["token_latency"].add(now - token.birth_time)
        if token.token_index > 0:
            self.stats["itl"].add(now - token.birth_time)
        if self.stats_cb is not None and token.sampled_for_stats:
            self.stats_cb(token, now)
        arrival, _, _ = self.requests[token.request_id]
        if token.token_index + 1 == self.tokens_per_request:
            self.completed_requests += 1
            if self.observer is not None:
                self.observer.request_completed(self, token.request_id)
            self.stats["request_latency"].add(now - arrival)
            del self.requests[token.request_id]
            self.admit()
        else:
            self.spawn_token(token.request_id)

    def should_capture_schedule_details(self, start, end):
        return (len(self.schedule_trace) < self.trace_limit
                or (self.debug and self.saturation_time is not None)
                or (self.observer is not None and getattr(self.observer, "needs_schedule_details", True))
                or (self.queue_trace is not None and self.queue_trace.observes_schedule(start, end)))

    def capture_schedule(self, trace):
        self.schedule_count += 1
        if self.observer is not None:
            self.observer.scheduled(trace)
        if self.queue_trace is not None:
            self.queue_trace.capture_schedule(trace)
        if len(self.schedule_trace) < self.trace_limit:
            self.schedule_trace.append(trace)
        if self.debug and self.saturation_time is not None:
            self.last_debug = trace

    def run(self):
        expected = self.total_requests * self.tokens_per_request
        while self.env.peek() < math.inf:
            if self.completed_tokens == expected:
                break
            if (self.stop_after is not None and self.saturation_time is not None
                    and self.completed_tokens - self.saturation_tokens >= self.stop_after):
                break
            if self.observer is not None:
                self.observer.advance(self, self.env.peek())
            self.env.step()
            self.steps += 1
        stopped = self.completed_tokens != expected
        if stopped and (self.stop_after is None or self.saturation_time is None):
            raise RuntimeError("event queue drained with unfinished requests")
        makespan = float(self.env.now)
        seconds = makespan / self.ticks_per_ms / 1000
        measures = {name: samples.summary(1 / self.ticks_per_ms) for name, samples in self.stats.items()}
        busy = [worker.observed_busy() for worker in self.workers]
        attn_hist, expert_hist, individual_hist, final_hist = Counter(), Counter(), Counter(), Counter()
        for worker in self.workers:
            attn_hist.update(worker.attention_batches)
            final_hist.update(worker.finalize_batches)
            expert_hist.update(worker.expert_batches)
            individual_hist.update(worker.expert_individual_batches)
        hist_mean = lambda hist: sum(k * n for k, n in hist.items()) / sum(hist.values()) if hist else 0.0
        tail_seconds = ((makespan - self.saturation_time) / self.ticks_per_ms / 1000
                        if self.saturation_time is not None else 0.0)
        unfinished_ready_ages = []
        for worker in self.workers:
            a, e, oldest = worker.snapshot()
            unfinished_ready_ages.extend(makespan - time for count, time in zip(a + e, oldest) if count)
        result = {
            "completion_times": self.completion_times,
            "avg_latency": self.stats["token_latency"].total / self.completed_tokens if self.completed_tokens else 0.0,
            "makespan": makespan, "makespan_ms": makespan / self.ticks_per_ms,
            "completed_tokens": self.completed_tokens, "completed_requests": self.completed_requests,
            "output_tokens_per_sec": self.completed_tokens / seconds if seconds else 0.0,
            "completed_requests_per_sec": self.completed_requests / seconds if seconds else 0.0,
            "avg_throughput_req_per_sec": self.completed_tokens / self.tokens_per_request / seconds if seconds else 0.0,
            "tail_throughput_req_per_sec": ((self.completed_tokens - self.saturation_tokens)
                / self.tokens_per_request / tail_seconds if tail_seconds else float("nan")),
            "avg_request_latency_ms": measures["request_latency"]["mean"] if self.completed_requests else float("nan"),
            "p90_request_latency_ms": measures["request_latency"]["p90"] if self.completed_requests else float("nan"),
            "p99_request_latency_ms": measures["request_latency"]["p99"] if self.completed_requests else float("nan"),
            "avg_per_expert_batch_size": hist_mean(individual_hist),
            "avg_expert_group_batch_size": hist_mean(expert_hist),
            "avg_attention_batch_size": hist_mean(attn_hist),
            "attention_batch_histogram": dict(attn_hist),
            "finalize_batch_histogram": dict(final_hist),
            "expert_group_batch_histogram": dict(expert_hist),
            "per_expert_batch_histogram": dict(individual_hist),
            "metrics_ms": measures,
            "gpu_compute_busy_fraction": [b / makespan if makespan else 0.0 for b, _ in busy],
            "gpu_compute_busy_ticks": [b for b, _ in busy],
            "gpu_stage_busy_ticks": [s for _, s in busy],
            "batch_overhead_model": self.overhead_model.describe(),
            "batch_overhead_accounting": {
                "scope": "sum of scheduled local batch estimates, including unfinished batches at early stop; not wall-time or observed CPU utilization",
                "by_gpu": [worker.overhead_accounting() for worker in self.workers],
            },
            "active_requests_at_end": len(self.requests), "waiting_requests_at_end": len(self.waiting),
            "unarrived_requests_at_end": self.total_requests - self.arrived_requests,
            "inflight_tokens_at_end": self.next_tid - self.completed_tokens,
            "incomplete_joins_at_end": len(self.joins),
            "inflight_network_branches_at_end": self.inflight_branches,
            "ready_attention_at_end": self.ready_attention, "ready_expert_at_end": self.ready_expert,
            "peak_active_requests": self.peak_active, "peak_waiting_requests": self.peak_waiting,
            "peak_incomplete_joins": self.peak_joins, "peak_network_branches": self.peak_inflight_branches,
            "peak_ready_attention": self.peak_ready_attention, "peak_ready_expert": self.peak_ready_expert,
            "oldest_ready_age_at_end_ms": max(unfinished_ready_ages, default=0.0) / self.ticks_per_ms,
            "remote_transfer_count": self.remote_transfers, "remote_expert_copy_count": self.remote_copies,
            "stopped_early": stopped, "steps_executed": self.steps,
            "saturation_step": self.saturation_step, "saturation_time": self.saturation_time,
            "saturation_tokens_completed": self.saturation_tokens,
            "scheduling_policy": self.policy,
            "expert_scheduling": self.expert_scheduling,
            "schedule_trace_layout": {
                "queue_order": ("A_0..A_L, then E_(layer,local_expert) in layer-major order"
                                if self.expert_scheduling == "expert" else "A_0..A_L, then E_0..E_(L-1)"),
                "num_attention_queues": self.num_layers + 1,
                "experts_per_gpu": self.experts_per_gpu,
                "expert_id_mapping": "gpu * experts_per_gpu + local_expert_id",
                "snapshot_units": "attention tokens; expert branches, before selection",
            },
            "schedule_trace": self.schedule_trace, "schedule_count": self.schedule_count,
            "schedule_trace_truncated": self.schedule_count > len(self.schedule_trace),
            "defrag_v0_debug_triggered": self.debug and self.saturation_time is not None,
            "defrag_v0_debug_trigger_time": self.saturation_time if self.debug else None,
            "defrag_v0_debug_logged": self.last_debug is not None,
            "defrag_v0_debug_entry": self.last_debug,
            "attention_cost_model": self.attention_cost.describe(),
            "cost_model": {
                "expert": "exact_grouped_profile" if self.expert_cost.grouped_lookup is not None else "sum_per_expert_proxy",
                "expert_scale": self.expert_cost.scale,
                "attention": self.attention_cost.describe()["kind"],
                "initial_context_length": self.initial_context,
                "finalize_per_token_ticks": self.finalize_t,
                "network": "dual_receive_fifo" if self.net_delay_fn is not None else "fixed_delay_no_contention",
                "hardware_calibrated": False,
            },
        }
        if self.queue_trace is not None:
            result["queue_trace"] = self.queue_trace.finish(makespan)
        if self.observer is not None:
            result["steady_metrics"] = self.observer.finish(self, fully_drained=not stopped)
        if self.token_trace is not None:
            result["token_trace"] = self.token_trace.finish(makespan, fully_drained=not stopped)
        if not self.quiet:
            print(f"{self.policy}: completed {self.completed_tokens}/{expected} tokens, "
                  f"{self.completed_requests}/{self.total_requests} requests, "
                  f"{result['output_tokens_per_sec']:.3f} tokens/s; "
                  f"ITL P95={measures['itl']['p95']} ms")
        return result


def run_simulation(
    ep_group_size=EP_GROUP_SIZE, total_expert_count=TOTAL_EXPERT_COUNT,
    max_batch_size=MAX_BATCH_SIZE, num_layers=NUM_LAYERS,
    total_requests=TOTAL_REQUESTS, tokens_per_request=TOKENS_PER_REQUEST,
    arrival_rate=ARRIVAL_RATE, expert_profile_path=EXPERT_COMPUTE_PROFILE_PATH,
    profile_routing_path=PROFILE_ROUTING_PATH, routing_top_k=ROUTING_TOP_K,
    global_request_max_batch_size=GLOBAL_REQUEST_MAX_BATCH_SIZE,
    attn_dp_group_size=ATTN_DP_GROUP_SIZE, tokens_after_reaching_max_bs=None,
    enable_defrag_v0_debug_log=False, scheduling_policy=SCHEDULING_POLICY,
    per_token_begin_cb=None, per_token_stats_cb=None, net_delay_fn=None,
    n_gpu_per_host=None, *, max_attention_batch_size=160,
    attention_profile_path=None, grouped_profile_path=None, expert_cost_scale=1.0,
    initial_context_length=0, finalize_per_token_t=0.0,
    attn_service_t=None, net_t_attn_to_expert=None, net_t_expert_to_attn=None,
    ticks_per_millisecond=TICKS_PER_MILLISECOND, rng_seed=RNG_SEED,
    defrag_lookahead_steps=4, defrag_lookback_steps=4, defrag_weight_decay=0.8,
    metric_sample_capacity=100000, schedule_trace_limit=0, quiet=False,
    queue_trace_path=None, queue_trace_interval_ms=0.1,
    queue_trace_start_ms=0.0, queue_trace_end_ms=None,
    arrival_times_ms=None, observer=None, routing_request_indices=None,
    expert_scheduling="layer", execution_mode="async", overhead_config=None,
    hidden_dim=2048, bytes_per_element=2,
    token_trace_path=None, token_trace_start_ms=0.0, token_trace_end_ms=None,
):
    """Run one finite decode workload; callbacks and legacy result keys retained.

    Costs/arrival_rate use ticks. Admission is a request-count cap, not KV-memory
    admission. Initial KV context plus token_index gives the decode KV history.
    First-token samples are excluded from metrics_ms['itl']. Receive contention
    is enabled only when net_delay_fn is supplied. Local transfers cost zero.
    Optional queue traces integrate ready queues over real event intervals;
    their window/interval parameters are milliseconds and add no sample events.
    Explicit arrival times are absolute milliseconds and override arrival_rate.
    Optional routing indices are profile request ordinals, with no token wrap.
    expert_scheduling='expert' compares individual expert queues with attention
    queues, executing only the winning expert up to max_batch_size. The default
    'layer' retains local grouped execution. Both share one GPU compute resource.
    Optional token traces sample attention selections in a millisecond window
    and follow each token through expert fork/join to its next attention start,
    even beyond the window. They add no simulator events and support async only.
    """
    positive_ints = dict(ep_group_size=ep_group_size, total_expert_count=total_expert_count,
        max_batch_size=max_batch_size, num_layers=num_layers, tokens_per_request=tokens_per_request,
        global_request_max_batch_size=global_request_max_batch_size,
        max_attention_batch_size=max_attention_batch_size, routing_top_k=routing_top_k,
        metric_sample_capacity=metric_sample_capacity, hidden_dim=hidden_dim,
        bytes_per_element=bytes_per_element)
    for name, value in positive_ints.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in dict(total_requests=total_requests, initial_context_length=initial_context_length,
                            schedule_trace_limit=schedule_trace_limit).items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if total_expert_count % ep_group_size or routing_top_k > total_expert_count:
        raise ValueError("experts must divide evenly across GPUs and top-k must not exceed experts")
    if attn_dp_group_size != ep_group_size:
        raise ValueError("unified compute requires attn_dp_group_size == ep_group_size")
    if tokens_after_reaching_max_bs is not None and (not isinstance(tokens_after_reaching_max_bs, Integral)
                                                   or tokens_after_reaching_max_bs < 0):
        raise ValueError("tokens_after_reaching_max_bs must be a nonnegative integer")
    if not math.isfinite(ticks_per_millisecond) or ticks_per_millisecond <= 0:
        raise ValueError("ticks_per_millisecond must be finite and positive")
    dispatch_t = NET_T_ATTN_TO_EXPERT if net_t_attn_to_expert is None else net_t_attn_to_expert
    return_t = NET_T_EXPERT_TO_ATTN if net_t_expert_to_attn is None else net_t_expert_to_attn
    for name, value in dict(arrival_rate=arrival_rate, dispatch_delay=dispatch_t,
                            return_delay=return_t, finalize_per_token_t=finalize_per_token_t).items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    gpus_per_host = ep_group_size if n_gpu_per_host is None else n_gpu_per_host
    if not isinstance(gpus_per_host, Integral) or gpus_per_host < 1:
        raise ValueError("n_gpu_per_host must be a positive integer")
    policy = "mtfs" if scheduling_policy == "longest_queue_first" else scheduling_policy
    if execution_mode not in ("async", "sync", "tbo"):
        raise ValueError("execution_mode must be 'async', 'sync', or 'tbo'")
    if token_trace_path is not None and execution_mode != "async":
        raise ValueError("token tracing supports execution_mode='async' only")
    # Reject invalid cost assumptions before loading profiles or opening files.
    overhead_config = BatchOverheadModel(
        ticks_per_millisecond=ticks_per_millisecond, **(overhead_config or {})).config()
    if expert_scheduling not in ("layer", "expert"):
        raise ValueError("expert_scheduling must be 'layer' or 'expert'")
    # Validate policy before loading potentially large profiles.
    if execution_mode == "async":
        UnifiedScheduler(num_layers, routing_top_k, policy, defrag_lookahead_steps,
                         defrag_lookback_steps, defrag_weight_decay)
    else:
        if expert_scheduling != "layer":
            raise ValueError("collective baselines require expert_scheduling='layer'")
        policy = execution_mode
    router = build_profile_router(profile_routing_path, total_expert_count, routing_top_k)
    router.validate_layers(num_layers)
    lookup = expert_compute_time_lookup_table_from_profile(expert_profile_path, max_batch_size,
                                                            ticks_per_millisecond=ticks_per_millisecond)
    expert_cost = ExpertCostModel(lookup, expert_cost_scale, grouped_profile_path, ticks_per_millisecond)
    attention_cost = AttentionCostModel(ATTN_SERVICE_T if attn_service_t is None else attn_service_t,
                                       attention_profile_path, ticks_per_millisecond)
    settings = dict(num_gpus=ep_group_size, num_experts=total_expert_count, num_layers=num_layers,
        top_k=routing_top_k, expert_batch_cap=max_batch_size, attn_batch_cap=max_attention_batch_size,
        total_requests=total_requests, tokens_per_request=tokens_per_request, arrival_rate=arrival_rate,
        active_cap=global_request_max_batch_size, initial_context=initial_context_length,
        finalize_t=finalize_per_token_t, dispatch_t=dispatch_t, return_t=return_t,
        net_delay_fn=net_delay_fn, gpus_per_host=gpus_per_host, ticks_per_ms=ticks_per_millisecond,
        seed=rng_seed, policy=policy, expert_scheduling=expert_scheduling, lookahead_steps=defrag_lookahead_steps,
        lookback_steps=defrag_lookback_steps, weight_decay=defrag_weight_decay,
        metric_sample_cap=metric_sample_capacity, trace_limit=schedule_trace_limit,
        stop_after=tokens_after_reaching_max_bs, debug=enable_defrag_v0_debug_log,
        queue_trace_path=queue_trace_path, queue_trace_interval_ms=queue_trace_interval_ms,
        queue_trace_start_ms=queue_trace_start_ms, queue_trace_end_ms=queue_trace_end_ms,
        token_trace_path=token_trace_path, token_trace_start_ms=token_trace_start_ms,
        token_trace_end_ms=token_trace_end_ms,
        arrival_times_ms=arrival_times_ms, observer=observer,
        routing_request_indices=routing_request_indices,
        overhead_config=overhead_config,
        hidden_dim=hidden_dim, bytes_per_element=bytes_per_element,
        begin_cb=per_token_begin_cb, stats_cb=per_token_stats_cb, quiet=quiet)
    simulator_type = AsyncMoESimulator
    if execution_mode != "async":
        if __package__:
            from .sim_collective import CollectiveMoESimulator
        else:
            from sim_collective import CollectiveMoESimulator
        simulator_type = CollectiveMoESimulator
        settings["execution_mode"] = execution_mode
    simulator = simulator_type(settings=settings, router=router, expert_cost=expert_cost,
                               attention_cost=attention_cost)
    try:
        return simulator.run()
    finally:
        try:
            if simulator.queue_trace is not None:
                simulator.queue_trace.close()
        finally:
            if simulator.token_trace is not None:
                simulator.token_trace.close(simulator.env.now)


if __name__ == "__main__":
    run_simulation()
