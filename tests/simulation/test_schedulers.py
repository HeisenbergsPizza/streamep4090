import math
import unittest

from simulation.schedulers import UnifiedScheduler


class UnifiedSchedulerTest(unittest.TestCase):
    def test_defrag_lookahead_can_choose_expert_over_largest_immediate_queue(self):
        scheduler = UnifiedScheduler(2, 2, lookahead_steps=2, lookback_steps=0,
                                     weight_decay=0.5)
        self.assertEqual(scheduler.select([4, 5], [8, 0]), ("expert", 0))
        self.assertEqual(scheduler.last_snapshot, [4, 5, 8, 0])
        self.assertEqual(scheduler.last_scores, [6.0, 5.0, 6.5, 0.0])

    def test_defrag_history_of_unselected_expert_changes_next_decision(self):
        scheduler = UnifiedScheduler(2, 2, lookahead_steps=2)
        self.assertEqual(scheduler.select([100, 0], [20, 0]), ("attention", 0))
        self.assertEqual(scheduler.select([1, 5], [0, 0]), ("attention", 0))
        self.assertEqual(scheduler.last_scores, [9.0, 5.0, 0.0, 0.0])
        no_history = UnifiedScheduler(2, 2, lookahead_steps=2, lookback_steps=0)
        self.assertEqual(no_history.select([1, 5], [0, 0]), ("attention", 1))

    def test_history_uses_bounded_mean_including_zero_snapshots(self):
        scheduler = UnifiedScheduler(2, 2, lookahead_steps=2, lookback_steps=2,
                                     weight_decay=1.0)
        scheduler.select([100, 0], [20, 0])  # E0 history: 10
        scheduler.select([100, 0], [0, 0])   # E0 history: 10, 0
        self.assertEqual(scheduler.select([1, 5], [0, 0]), ("attention", 0))
        self.assertEqual(scheduler.last_scores[0], 6.0)
        # The first snapshot has now been evicted: E0 history is 0, 0.
        self.assertEqual(scheduler.select([1, 5], [0, 0]), ("attention", 1))
        self.assertEqual(scheduler.last_scores[0], 1.0)

    def test_expert_history_is_truncated_before_averaging(self):
        scheduler = UnifiedScheduler(2, 4, lookahead_steps=2)
        scheduler.select([100, 0], [7, 0])  # effective E0=1.75, stored as 1
        self.assertEqual(scheduler.select([1, 2], [0, 0]), ("attention", 1))
        self.assertAlmostEqual(scheduler.last_scores[0], 1.8)

    def test_idle_calls_do_not_erase_history_or_select_stale_work(self):
        scheduler = UnifiedScheduler(2, 2, lookahead_steps=2, lookback_steps=1)
        scheduler.select([100, 0], [20, 0])
        for _ in range(3):
            self.assertIsNone(scheduler.select([0, 0], [0, 0]))
            self.assertEqual(scheduler.last_snapshot, [0, 0, 0, 0])
            self.assertEqual(scheduler.last_scores, [0.0] * 4)
        self.assertEqual(scheduler.select([1, 5], [0, 0]), ("attention", 0))

    def test_ties_use_unified_index_while_flfs_uses_pipeline_order(self):
        for policy in ("defrag", "defragging_v0", "mtfs"):
            with self.subTest(policy=policy):
                scheduler = UnifiedScheduler(2, 2, policy, lookahead_steps=1)
                self.assertEqual(scheduler.select([0, 4], [8, 0]), ("attention", 1))
        scheduler = UnifiedScheduler(2, 2, "flfs")
        self.assertEqual(scheduler.select([0, 4], [8, 0]), ("expert", 0))

    def test_mtfs_compares_effective_counts_not_expert_copies(self):
        scheduler = UnifiedScheduler(1, 4, "mtfs")
        self.assertEqual(scheduler.select([3], [8]), ("attention", 0))
        self.assertEqual(scheduler.select([3], [16]), ("expert", 0))

    def test_fifo_compares_ready_times_and_breaks_ties_by_unified_index(self):
        scheduler = UnifiedScheduler(2, 2, "fifo")
        self.assertEqual(scheduler.select([0, 4], [8, 1],
                                         [math.inf, 5.0, 3.0, 7.0]), ("expert", 0))
        self.assertEqual(scheduler.select([0, 4], [8, 1],
                                         [math.inf, 3.0, 3.0, 7.0]), ("attention", 1))
        self.assertIsNone(scheduler.select([0, 0], [0, 0]))

    def test_extra_attention_follows_last_expert_and_wraps_to_first_attention(self):
        scheduler = UnifiedScheduler(1, 1, lookahead_steps=2, lookback_steps=0,
                                     weight_decay=0.5)
        # Pipeline A0, E0, A1; unified storage A0, A1, E0.
        self.assertEqual(scheduler.select([1, 4], [3]), ("expert", 0))
        self.assertEqual(scheduler.last_scores, [2.5, 4.5, 5.0])
        self.assertEqual(scheduler.select([0, 4], [0]), ("attention", 1))
        flfs = UnifiedScheduler(1, 1, "flfs")
        self.assertEqual(flfs.select([0, 4], [3]), ("expert", 0))

    def test_lookahead_counts_pipeline_stages_and_wraps_circularly(self):
        scheduler = UnifiedScheduler(1, 2, lookahead_steps=4, lookback_steps=0,
                                     weight_decay=0.5)
        self.assertEqual(scheduler.select([2], [2]), ("attention", 0))
        # A0: 2 + .5*1 + .25*2 + .125*1; E0 analogous.
        self.assertEqual(scheduler.last_scores, [3.125, 2.5])

    def test_snapshot_does_not_alias_callers_counts(self):
        attn, expert = [2], [1]
        scheduler = UnifiedScheduler(1, 1)
        scheduler.select(attn, expert)
        attn[0] = 0
        expert[0] = 0
        self.assertEqual(scheduler.last_snapshot, [2, 1])

    def test_invalid_constructor_parameters(self):
        invalid = [
            {"num_layers": 0}, {"num_layers": 1.5}, {"top_k": 0},
            {"top_k": True}, {"policy": "unknown"}, {"lookahead_steps": 0},
            {"lookback_steps": -1}, {"weight_decay": -0.1},
            {"weight_decay": 1.1}, {"weight_decay": math.nan},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                kwargs = {"num_layers": 2, "top_k": 2, **overrides}
                with self.assertRaises(ValueError):
                    UnifiedScheduler(**kwargs)

    def test_invalid_queue_shapes_and_counts(self):
        scheduler = UnifiedScheduler(2, 2)
        for attn, expert in (([1], [1, 1]), ([1, 1], [1]),
                             ([1, 1, 1, 1], [1, 1]), ([-1, 0], [0, 0]),
                             ([1.5, 0], [0, 0]), ([True, 0], [0, 0])):
            with self.subTest(attn=attn, expert=expert):
                with self.assertRaises(ValueError):
                    scheduler.select(attn, expert)

    def test_fifo_rejects_missing_or_nonfinite_ready_time_for_work(self):
        scheduler = UnifiedScheduler(1, 1, "fifo")
        for ready in (None, [1.0], [math.inf, 0.0], [math.nan, 0.0], ["old", 0.0]):
            with self.subTest(ready=ready):
                with self.assertRaises(ValueError):
                    scheduler.select([1], [0], ready)


if __name__ == "__main__":
    unittest.main()
