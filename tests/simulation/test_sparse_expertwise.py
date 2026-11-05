"""Differential checks against the expert-wise algorithm before sparse optimization."""
from collections import deque
import math
import random
import unittest

from simulation.schedulers import ExpertWiseScheduler, UnifiedScheduler


class DenseReference(UnifiedScheduler):
    """Frozen dense score/selection oracle, independent of select_sparse."""

    def __init__(self, *args, experts_per_layer, **kwargs):
        super().__init__(*args, **kwargs)
        self.experts_per_layer = experts_per_layer

    def select(self, attention, experts, oldest=None):
        n = len(attention)
        stage_oldest = (None if oldest is None else list(oldest[:n]) + [
            min((oldest[n + layer * self.experts_per_layer + expert]
                 for expert, count in enumerate(row) if count), default=math.inf)
            for layer, row in enumerate(experts)])
        super().select(attention, [sum(row) for row in experts], stage_oldest)
        raw = list(attention) + [count for row in experts for count in row]
        scores = self.last_scores[:n]
        for layer, row in enumerate(experts):
            for expert, count in enumerate(row):
                score = 0.0
                if count:
                    if self.policy in ("defrag", "mtfs"):
                        score = count / self.top_k
                        for contribution in self._lookahead_contributions[n + layer]:
                            score += contribution
                    elif self.policy == "flfs":
                        score = self.last_scores[n + layer]
                    else:
                        score = -float(oldest[n + layer * self.experts_per_layer + expert])
                scores.append(score)
        self.last_snapshot, self.last_scores = raw, scores
        candidates = [index for index, count in enumerate(raw) if count]
        self.last_selected_index = max(candidates, key=lambda index: scores[index]) if candidates else None
        index = self.last_selected_index
        if index is None:
            return None
        if index < n:
            return "attention", index, None
        layer, expert = divmod(index - n, self.experts_per_layer)
        return "expert", layer, expert


class SparseExpertWiseTests(unittest.TestCase):
    def test_random_sequences_match_frozen_dense_oracle_exactly(self):
        # W extends beyond one pipeline revolution; toggled final A checks
        # history's fixed index mapping. Rounded timestamps create FIFO ties.
        for policy in ("defrag", "defragging_v0", "flfs", "mtfs", "fifo"):
            for lookback in (0, 4):
                with self.subTest(policy=policy, lookback=lookback):
                    rng = random.Random(714)
                    kwargs = dict(num_layers=4, top_k=3, policy=policy, lookahead_steps=13,
                                  lookback_steps=lookback, weight_decay=.8, experts_per_layer=5)
                    expected, dense, sparse = DenseReference(**kwargs), ExpertWiseScheduler(**kwargs), ExpertWiseScheduler(**kwargs)
                    for step in range(100):
                        n = 4 + step % 2
                        attention = [rng.randrange(12) if rng.random() < .3 else 0 for _ in range(n)]
                        experts = [[rng.randrange(15) if rng.random() < .25 else 0 for _ in range(5)] for _ in range(4)]
                        if step % 11 == 0:
                            attention, experts = [0] * n, [[0] * 5 for _ in range(4)]
                        raw = attention + [count for row in experts for count in row]
                        oldest = [float(rng.randrange(5)) if count else math.inf for count in raw]
                        # Also exercise no timestamps, which all non-FIFO policies permit.
                        if policy != "fifo" and step % 3 == 0:
                            oldest = None
                        entries = [(layer, expert, count) for layer, row in enumerate(experts)
                                   for expert, count in enumerate(row) if count]
                        rng.shuffle(entries)
                        sparse_oldest = (None if oldest is None else oldest[:n] + [
                            oldest[n + layer * 5 + expert] for layer, expert, _ in entries])
                        want = expected.select(attention, experts, oldest)
                        self.assertEqual(dense.select(attention, experts, oldest), want)
                        details = step % 3 == 0
                        self.assertEqual(sparse.select_sparse(attention, entries, sparse_oldest, record_details=details), want)
                        self.assertEqual(sparse.last_selected_index, expected.last_selected_index)
                        self.assertEqual(list(sparse._history), list(expected._history))
                        if not details:
                            self.assertEqual(sparse.last_snapshot, [])
                            self.assertEqual(sparse.last_scores, [])
                        history_before = list(sparse._history)
                        sparse.materialize_details()
                        self.assertEqual(sparse.last_snapshot, expected.last_snapshot)
                        self.assertEqual(sparse.last_scores, expected.last_scores)
                        self.assertEqual(dense.last_snapshot, expected.last_snapshot)
                        self.assertEqual(dense.last_scores, expected.last_scores)
                        self.assertEqual(list(sparse._history), history_before)
                        # Materialization is repeatable and makes no scheduling decision.
                        self.assertEqual(sparse.materialize_details(), (expected.last_snapshot, expected.last_scores))

    def test_unsorted_sparse_fifo_timestamps_stay_attached_to_experts(self):
        scheduler = ExpertWiseScheduler(2, 1, "fifo", experts_per_layer=3)
        entries = [(1, 2, 9), (0, 2, 3), (0, 0, 1)]
        self.assertEqual(scheduler.select_sparse([0, 0], entries, [math.inf, math.inf, 1, 4, 2]),
                         ("expert", 1, 2))
        self.assertEqual(scheduler.last_scores, [0, 0, -2, 0, -4, 0, 0, -1])
        self.assertEqual(scheduler.select_sparse([0, 0], entries, [math.inf, math.inf, 1, 1, 1]),
                         ("expert", 0, 0))

    def test_lazy_details_survive_caller_mutation_and_idle_replaces_them(self):
        scheduler = ExpertWiseScheduler(2, 2, "mtfs", experts_per_layer=3)
        attention, entries = [1, 2], [[1, 2, 9]]
        scheduler.select_sparse(attention, entries, record_details=False)
        attention[:] = [8, 8]
        entries[0][2] = 99
        self.assertEqual(scheduler.materialize_details(), ([1, 2, 0, 0, 0, 0, 0, 9], [1, 2, 0, 0, 0, 0, 0, 4.5]))
        before = list(scheduler._history)
        self.assertIsNone(scheduler.select_sparse([0, 0, 0], [], record_details=False))
        self.assertEqual(list(scheduler._history), before)
        self.assertEqual(scheduler.materialize_details(), ([0] * 9, [0] * 9))
        self.assertIsNone(scheduler.last_selected_index)

    def test_bad_sparse_inputs_leave_history_unchanged(self):
        scheduler = ExpertWiseScheduler(2, 2, "fifo", experts_per_layer=3)
        scheduler.select_sparse([1, 0], [], [1, math.inf])
        before = list(scheduler._history)
        cases = [
            ([0, 0], [(0, 0, 0)], None),
            ([0, 0], [(0, 0, -1)], None),
            ([0, 0], [(0, 0, True)], None),
            ([0, 0], [(0, 0, 1.5)], None),
            ([0, 0], [(True, 0, 1)], None),
            ([0, 0], [(0, False, 1)], None),
            ([0, 0], [(2, 0, 1)], None),
            ([0, 0], [(0, 3, 1)], None),
            ([0, 0], [(0, 0)], None),
            ([0, 0], [(0, 0, 1), (0, 0, 2)], None),
            ([0], [(0, 0, 1)], [math.inf, 1]),
            ([0, 0], [(0, 0, 1)], None),
            ([0, 0], [(0, 0, 1)], [math.inf, math.inf]),
            ([0, 0], [(0, 0, 1)], [math.inf, math.inf, math.inf]),
            ([0, 0], [(0, 0, 1)], [math.inf, math.inf, math.nan]),
            ([0, 0], [(0, 0, 1)], [math.inf, math.inf, True]),
            ([0, 0], [(0, 0, 1)], [math.inf, math.inf, "old"]),
        ]
        for attention, entries, oldest in cases:
            with self.subTest(entries=entries, oldest=oldest), self.assertRaises(ValueError):
                scheduler.select_sparse(attention, entries, oldest, record_details=False)
            self.assertEqual(list(scheduler._history), before)

    def test_builtin_integer_fastpath_retains_bool_and_range_rejection(self):
        for value in (True, False, -1, 1.5):
            scheduler = ExpertWiseScheduler(1, 2, experts_per_layer=2)
            with self.subTest(value=value), self.assertRaises(ValueError):
                scheduler.select([0], [[value, 0]])


if __name__ == "__main__":
    unittest.main()
