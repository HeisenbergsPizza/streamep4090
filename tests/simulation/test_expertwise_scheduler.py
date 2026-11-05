"""Independent score examples and invariants for expert-granular choices."""
import math
import unittest

from simulation.schedulers import ExpertWiseScheduler, UnifiedScheduler


class ExpertWiseSchedulerTests(unittest.TestCase):
    def test_global_argmax_does_not_choose_largest_aggregate_layer_first(self):
        for policy in ("defrag", "mtfs"):
            scheduler = ExpertWiseScheduler(2, 1, policy, lookahead_steps=1,
                                            experts_per_layer=2)
            self.assertEqual(scheduler.select([0, 0], [[6, 6], [10, 0]]), ("expert", 1, 0))
            self.assertEqual(scheduler.last_scores, [0, 0, 6, 6, 10, 0])
            self.assertEqual(scheduler.last_selected_index, 4)

    def test_sibling_experts_share_downstream_bonus_and_compete_with_attention(self):
        scheduler = ExpertWiseScheduler(2, 2, lookahead_steps=2, lookback_steps=0,
                                        weight_decay=.5, experts_per_layer=2)
        # A0 sees aggregate E0=(10+2)/2=6. Each E0,e sees A1=5.
        self.assertEqual(scheduler.select([4, 5], [[10, 2], [0, 0]]), ("expert", 0, 0))
        self.assertEqual(scheduler.last_scores, [7, 5, 7.5, 3.5, 0, 0])
        self.assertEqual(scheduler.last_snapshot, [4, 5, 10, 2, 0, 0])
        # Relabeling peers only relabels their priorities; there is no e0 -> e1 edge.
        scheduler.select([4, 5], [[2, 10], [0, 0]])
        self.assertEqual(scheduler.last_scores, [7, 5, 3.5, 7.5, 0, 0])
        self.assertEqual(scheduler.last_selected_index, 3)

    def test_topk_normalization_applies_to_individual_immediate_work(self):
        scheduler = ExpertWiseScheduler(1, 4, "mtfs", experts_per_layer=2)
        self.assertEqual(scheduler.select([3], [[8, 8]]), ("attention", 0, None))
        self.assertEqual(scheduler.last_scores, [3, 2, 2])

    def test_history_truncates_aggregate_not_individuals_and_ignores_idle(self):
        scheduler = ExpertWiseScheduler(2, 4, lookahead_steps=2, lookback_steps=2,
                                        weight_decay=1, experts_per_layer=4)
        scheduler.select([100, 0], [[3, 3, 3, 3], [0, 0, 0, 0]])
        for _ in range(3):
            self.assertIsNone(scheduler.select([0, 0], [[0]*4, [0]*4]))
            self.assertIsNone(scheduler.last_selected_index)
        # Stored effective E0 = int(12/4) = 3, rather than 4*int(3/4)=0.
        self.assertEqual(scheduler.select([1, 3], [[0]*4, [0]*4]), ("attention", 0, None))
        self.assertEqual(scheduler.last_scores[:2], [4, 3])
        self.assertEqual(scheduler.select([1, 3], [[0]*4, [0]*4]), ("attention", 1, None))
        self.assertEqual(scheduler.last_scores[:2], [2.5, 3])
        self.assertTrue(all(score == 0 for score in scheduler.last_scores[2:]))
        self.assertEqual(scheduler.select([1, 3], [[0]*4, [0]*4]), ("attention", 1, None))
        self.assertEqual(scheduler.last_scores[:2], [1, 3])

    def test_fifo_uses_each_experts_own_timestamp_and_deterministic_ties(self):
        scheduler = ExpertWiseScheduler(1, 2, "fifo", experts_per_layer=2)
        self.assertEqual(scheduler.select([1], [[10, 1]], [5, 4, 2]), ("expert", 0, 1))
        self.assertEqual(scheduler.last_scores, [-5, -4, -2])
        self.assertEqual(scheduler.select([1], [[10, 1]], [2, 2, 2]), ("attention", 0, None))
        self.assertEqual(scheduler.select([0], [[10, 1]], [math.inf, 2, 2]), ("expert", 0, 0))

    def test_flfs_uses_pipeline_order_and_lower_expert_id_on_sibling_ties(self):
        scheduler = ExpertWiseScheduler(2, 2, "flfs", experts_per_layer=2)
        self.assertEqual(scheduler.select([0, 100], [[1, 10], [100, 0]]), ("expert", 0, 0))
        self.assertEqual(scheduler.select([1, 100], [[1, 10], [100, 0]]), ("attention", 0, None))

    def test_one_local_expert_is_exactly_equivalent_including_finalize_and_history(self):
        snapshots = [([1, 4, 2], [7, 3]), ([0, 0, 0], [0, 0]),
                     ([1, 0, 2], [3, 0]), ([0, 4], [2, 7]), ([1, 0, 1], [0, 8])]
        for policy in ("defrag", "defragging_v0", "mtfs", "flfs", "fifo"):
            layer = UnifiedScheduler(2, 2, policy, lookahead_steps=7, lookback_steps=3)
            expert = ExpertWiseScheduler(2, 2, policy, lookahead_steps=7, lookback_steps=3,
                                         experts_per_layer=1)
            for attn, counts in snapshots:
                oldest = [i + .5 if n else math.inf for i, n in enumerate(attn + counts)]
                want = layer.select(attn, counts, oldest)
                got = expert.select(attn, [[n] for n in counts], oldest)
                self.assertEqual(got, None if want is None else (*want, 0 if want[0] == "expert" else None))
                self.assertEqual(expert.last_scores, layer.last_scores)
                self.assertEqual(expert.last_snapshot, layer.last_snapshot)
                self.assertEqual(expert.last_selected_index, layer.last_selected_index)

    def test_invalid_shapes_counts_and_timestamps_fail_before_history_changes(self):
        scheduler = ExpertWiseScheduler(1, 2, "fifo", experts_per_layer=2)
        for attn, counts, oldest in [([0], [[1]], None), ([0], [[-1, 0]], None),
                                     ([0], [[True, 0]], None), ([0], [[1.5, 0]], None),
                                     ([0], [[1, 1]], [math.inf, 0, math.inf]),
                                     ([0], [[1, 1]], [math.inf, 0, math.nan]),
                                     ([0], [[1, 1]], [math.inf, 0]),
                                     ([0], [[1, 1]], [math.inf, 0, "old"]),
                                     ([0, 0, 0], [[1, 1]], None)]:
            with self.subTest(counts=counts, oldest=oldest), self.assertRaises(ValueError):
                scheduler.select(attn, counts, oldest)
        self.assertFalse(scheduler._history)
        with self.assertRaises(ValueError):
            ExpertWiseScheduler(1, 1, experts_per_layer=0)


if __name__ == "__main__":
    unittest.main()
