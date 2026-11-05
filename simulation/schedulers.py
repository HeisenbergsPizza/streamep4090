"""Pure-Python policies for a GPU sharing attention and expert compute.

The layer-granular defrag policy follows ``UnifiedDefraggingLayerScheduler`` in
``csrc/muhelper/layer.cpp``. Its queue snapshots and scores use the C++ unified
index order: all attention queues, followed by all expert queues. Lookahead
instead follows the circular execution pipeline A0, E0, A1, E1, ... .
ExpertWiseScheduler extends the candidate set while retaining stage lookahead.
"""

from __future__ import annotations

import math
from collections import deque
from numbers import Integral, Real
from typing import Literal


Stage = Literal["attention", "expert"]


def _integer(name: str, value: int, minimum: int) -> int:
    # Queue lengths are native ints. Avoid the numbers.Integral ABC machinery
    # for this hot path while retaining the public checks for other scalars.
    if type(value) is int:
        if value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return value
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class UnifiedScheduler:
    """Select one ready attention/expert layer without modifying its queues.

    ``num_layers`` is the number of expert layers. Attention may have either
    that many queues or one extra final queue, which follows the last expert
    stage in the pipeline. Expert queue counts include top-k copies and are
    divided by ``top_k`` when comparing effective work.

    ``lookahead_steps`` includes the immediate stage: 1 disables lookahead.
    Defrag adds decayed downstream current counts and their historical mean;
    history stores integer-truncated effective counts for the most recent
    ``lookback_steps`` successful selections, including unselected queues.
    Idle calls do not advance history.

    FIFO requires ``oldest_ready`` in the same [A..., E...] order as the
    counts. Nonempty queues require finite timestamps; empty queues may use
    infinity. Equal scores or FIFO timestamps prefer the smallest unified
    index (all A slots before all E slots). FLFS uses pipeline order instead.

    ``last_snapshot`` is a raw pre-selection count list and ``last_scores``
    contains policy priorities in unified order (larger is preferred). Empty
    queues have score zero and are never candidates. FIFO priorities are the
    negated timestamps. These lists are replaced on each valid call.
    """

    def __init__(
        self,
        num_layers: int,
        top_k: int,
        policy: str = "defrag",
        lookahead_steps: int = 4,
        lookback_steps: int = 4,
        weight_decay: float = 0.8,
    ) -> None:
        self.num_layers = _integer("num_layers", num_layers, 1)
        self.top_k = _integer("top_k", top_k, 1)
        self.lookahead_steps = _integer("lookahead_steps", lookahead_steps, 1)
        self.lookback_steps = _integer("lookback_steps", lookback_steps, 0)
        if (
            isinstance(weight_decay, bool)
            or not isinstance(weight_decay, Real)
            or not math.isfinite(weight_decay)
            or not 0.0 <= weight_decay <= 1.0
        ):
            raise ValueError("weight_decay must be finite and between 0 and 1")
        self.weight_decay = float(weight_decay)
        if policy == "defragging_v0":
            policy = "defrag"
        if policy not in ("defrag", "flfs", "mtfs", "fifo"):
            raise ValueError(f"Unknown scheduling policy: {policy!r}")
        self.policy = policy
        self.last_snapshot: list[int] = []
        self.last_scores: list[float] = []
        self.last_selected_index: int | None = None
        self._lookahead_contributions: list[list[float]] = []
        # Reserve the optional final A slot so its presence cannot shift E
        # history indices between calls.
        self._history: deque[tuple[int, ...]] = deque(maxlen=self.lookback_steps)

    def select(
        self,
        attn_counts: list[int],
        expert_counts: list[int],
        oldest_ready: list[float] | None = None,
    ) -> tuple[Stage, int] | None:
        if len(attn_counts) not in (self.num_layers, self.num_layers + 1):
            raise ValueError("attn_counts must have num_layers or num_layers + 1 entries")
        if len(expert_counts) != self.num_layers:
            raise ValueError("expert_counts must have num_layers entries")
        raw = [
            _integer("queue count", count, 0)
            for count in list(attn_counts) + list(expert_counts)
        ]
        num_attn = len(attn_counts)
        candidates = [i for i, count in enumerate(raw) if count > 0]
        if oldest_ready is not None:
            if len(oldest_ready) != len(raw):
                raise ValueError("oldest_ready must have one entry per unified queue")
            if any(
                isinstance(timestamp, bool) or not isinstance(timestamp, Real)
                for timestamp in oldest_ready
            ):
                raise ValueError("oldest_ready entries must be numeric timestamps")
        if self.policy == "fifo" and candidates:
            if oldest_ready is None or any(
                not math.isfinite(oldest_ready[i]) for i in candidates
            ):
                raise ValueError("fifo requires a finite oldest_ready timestamp for every nonempty queue")

        self.last_snapshot = raw
        self.last_scores = [0.0] * len(raw)
        self.last_selected_index = None
        self._lookahead_contributions = [[] for _ in raw]
        if not candidates:
            return None

        effective = [float(count) for count in raw[:num_attn]] + [
            count / self.top_k for count in raw[num_attn:]
        ]
        pipeline = [
            index
            for layer in range(self.num_layers)
            for index in (layer, num_attn + layer)
        ]
        if num_attn > self.num_layers:
            pipeline.append(self.num_layers)
        positions = {index: pos for pos, index in enumerate(pipeline)}

        if self.policy == "defrag":
            history_means = [0.0] * len(raw)
            if self._history:
                for index in range(len(raw)):
                    history_index = (
                        index if index < num_attn
                        else self.num_layers + 1 + index - num_attn
                    )
                    history_means[index] = (
                        sum(snapshot[history_index] for snapshot in self._history)
                        / len(self._history)
                    )
            for index in candidates:
                score = effective[index]
                decay = self.weight_decay
                for offset in range(1, self.lookahead_steps):
                    downstream = pipeline[(positions[index] + offset) % len(pipeline)]
                    contribution = (effective[downstream] + history_means[downstream]) * decay
                    self._lookahead_contributions[index].append(contribution)
                    score += contribution
                    decay *= self.weight_decay
                self.last_scores[index] = score
        elif self.policy == "mtfs":
            self.last_scores = effective
        elif self.policy == "flfs":
            for index in candidates:
                self.last_scores[index] = float(len(pipeline) - positions[index])
        else:
            assert oldest_ready is not None
            for index in candidates:
                self.last_scores[index] = -float(oldest_ready[index])

        # candidates retain unified order; max returns its first tied entry.
        best = max(candidates, key=lambda index: self.last_scores[index])
        self.last_selected_index = best
        history_snapshot = [int(count) for count in effective[:num_attn]]
        if num_attn == self.num_layers:
            history_snapshot.append(0)
        history_snapshot.extend(int(count) for count in effective[num_attn:])
        self._history.append(tuple(history_snapshot))
        if best < num_attn:
            return "attention", best
        return "expert", best - num_attn


class ExpertWiseScheduler(UnifiedScheduler):
    """Compare attention queues with individual local expert queues globally.

    This is a simulation extension of the unified scheduler, not a reproduction
    of the author's unfinished expert-wise path. Immediate expert work is
    Q(layer, expert) / top_k. Defrag lookahead and history retain the base
    scheduler's *stage aggregates*: sum_e Q(layer, e) / top_k. Sibling experts
    are parallel branches, never successive lookahead stages. Consequently
    siblings share the downstream bonus but have independent immediate scores.

    Counts have shape [layer][local expert]; oldest_ready, last_snapshot and
    last_scores use [A..., E_0,0, E_0,1, ..., E_1,0, ...]. The result is
    (stage, layer, local_expert), with None for attention. Ties prefer this
    flattened order; FLFS ties between siblings prefer the lower expert ID.
    History advances exactly once per nonempty selection, before any drain,
    and stores int(sum_e Q / top_k), not sum_e int(Q / top_k).
    """

    def __init__(self, num_layers, top_k, policy="defrag", lookahead_steps=4,
                 lookback_steps=4, weight_decay=0.8, *, experts_per_layer):
        super().__init__(num_layers, top_k, policy, lookahead_steps,
                         lookback_steps, weight_decay)
        self.experts_per_layer = _integer("experts_per_layer", experts_per_layer, 1)
        self._sparse_details = None

    def select(self, attn_counts, expert_counts, oldest_ready=None):
        """Dense public interface; always materialize the full decision details."""
        if len(expert_counts) != self.num_layers or any(
            not isinstance(row, (list, tuple)) or len(row) != self.experts_per_layer
            for row in expert_counts
        ):
            raise ValueError("expert_counts must have shape [num_layers][experts_per_layer]")
        experts = [[_integer("queue count", count, 0) for count in row]
                   for row in expert_counts]
        num_attn = len(attn_counts)
        if oldest_ready is not None:
            if len(oldest_ready) != num_attn + self.num_layers * self.experts_per_layer:
                raise ValueError("oldest_ready must have one entry per attention/expert queue")
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in oldest_ready):
                raise ValueError("oldest_ready entries must be numeric timestamps")
        entries = [(layer, expert, count) for layer, row in enumerate(experts)
                   for expert, count in enumerate(row) if count]
        sparse_oldest = (None if oldest_ready is None else list(oldest_ready[:num_attn]) + [
            oldest_ready[num_attn + layer * self.experts_per_layer + expert]
            for layer, expert, _ in entries])
        return self.select_sparse(attn_counts, entries, sparse_oldest)

    def select_sparse(self, attn_counts, expert_entries, oldest_ready=None, *, record_details=True):
        """Select with only nonempty expert queues, preserving dense semantics.

        ``expert_entries`` contains unique ``(layer, local_expert, count)``
        triples with positive counts, in any order. If supplied, timestamps use
        ``[all attention queues, experts in input-entry order]``. FIFO needs
        finite timestamps for nonempty queues; other policies need no times.

        With ``record_details=False``, last_snapshot and last_scores are empty
        until materialize_details() is called. Selection, tie-breaking and
        history are unchanged, and last_selected_index is always a dense index.
        The saved sparse details belong to this call, independent of later
        caller mutation. Full details cost O(layers * experts) only on demand.
        """
        num_attn = len(attn_counts)
        entries, seen = [], set()
        for item in expert_entries:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError("expert_entries must contain (layer, local_expert, positive count) triples")
            layer, expert, count = (_integer("expert layer", item[0], 0),
                                    _integer("local expert", item[1], 0),
                                    _integer("queue count", item[2], 1))
            if layer >= self.num_layers or expert >= self.experts_per_layer:
                raise ValueError("expert entry layer or local expert is out of range")
            if (layer, expert) in seen:
                raise ValueError("expert_entries must contain unique expert queues")
            seen.add((layer, expert))
            entries.append((layer, expert, count))
        if oldest_ready is not None:
            if len(oldest_ready) != num_attn + len(entries):
                raise ValueError("oldest_ready must have one entry per attention/sparse expert queue")
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in oldest_ready):
                raise ValueError("oldest_ready entries must be numeric timestamps")
        if self.policy == "fifo" and (any(attn_counts) or entries):
            if oldest_ready is None or any(
                not math.isfinite(oldest_ready[i])
                for i, count in enumerate(list(attn_counts) + [item[2] for item in entries]) if count
            ):
                raise ValueError("fifo requires a finite oldest_ready timestamp for every nonempty queue")

        # Associate times before sorting: callers may maintain a set of ready
        # queues. Dense unified order is the deterministic tie-breaker.
        entries = sorted((layer, expert, count,
                          oldest_ready[num_attn + index] if oldest_ready is not None else None)
                         for index, (layer, expert, count) in enumerate(entries))
        stage_counts = [0] * self.num_layers
        stage_oldest = (list(oldest_ready[:num_attn]) + [math.inf] * self.num_layers
                        if oldest_ready is not None else None)
        for layer, _, count, timestamp in entries:
            stage_counts[layer] += count
            if stage_oldest is not None:
                stage_oldest[num_attn + layer] = min(stage_oldest[num_attn + layer], timestamp)
        # Reuse the exact stage lookahead/history calculation. Its provisional
        # winner is unused: the only dispatch choice is the global argmax below.
        super().select(attn_counts, stage_counts, stage_oldest)
        attn_snapshot, attn_scores = self.last_snapshot[:num_attn], self.last_scores[:num_attn]
        best, best_score = None, -math.inf
        for index, count in enumerate(attn_snapshot):
            if count and (best is None or attn_scores[index] > best_score):
                best, best_score = index, attn_scores[index]
        details = []
        for layer, expert, count, timestamp in entries:
            stage_index = num_attn + layer
            if self.policy in ("defrag", "mtfs"):
                score = count / self.top_k
                # Keep the addition order of the dense implementation exactly.
                for contribution in self._lookahead_contributions[stage_index]:
                    score += contribution
            elif self.policy == "flfs":
                score = self.last_scores[stage_index]
            else:
                score = -float(timestamp)
            index = num_attn + layer * self.experts_per_layer + expert
            details.append((index, count, score))
            if best is None or score > best_score:
                best, best_score = index, score
        self._sparse_details = (attn_snapshot, attn_scores, details)
        self.last_snapshot, self.last_scores = [], []
        self.last_selected_index = best
        if record_details:
            self.materialize_details()
        if best is None:
            return None
        if best < num_attn:
            return "attention", best, None
        layer, expert = divmod(best - num_attn, self.experts_per_layer)
        return "expert", layer, expert

    def materialize_details(self):
        """Populate and return this decision's (snapshot, scores), without history changes."""
        if self._sparse_details is not None:
            attn, attn_scores, experts = self._sparse_details
            self.last_snapshot = list(attn) + [0] * (self.num_layers * self.experts_per_layer)
            self.last_scores = list(attn_scores) + [0.0] * (self.num_layers * self.experts_per_layer)
            for index, count, score in experts:
                self.last_snapshot[index], self.last_scores[index] = count, score
        return self.last_snapshot, self.last_scores


__all__ = ["UnifiedScheduler", "ExpertWiseScheduler"]
