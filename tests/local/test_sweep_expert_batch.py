"""CPU checks for selecting expert batch sizes from available GPU memory."""

from pathlib import Path
import sys
import unittest


OPS = Path(__file__).resolve().parents[2] / "experiments/scripts/ops"
sys.path.insert(0, str(OPS))
try:
    import sweep_expert_batch as sweep
finally:
    sys.path.pop(0)


class ExpertBatchSweepTest(unittest.TestCase):
    def test_sweep_includes_the_endpoint_once(self):
        for limit, expected in ((1, [1]), (4, [1, 2, 4]), (7, [1, 2, 4, 7])):
            with self.subTest(limit=limit):
                self.assertEqual(list(sweep.batch_sizes(limit)), expected)

    def test_memory_budget_reserves_weights_and_rounds_down(self):
        # H=128, I=64 uses 49,152 weight bytes and 900 bytes per batch row.
        # 80% of 67,065 free bytes fits weights plus exactly five rows.
        for free_bytes, expected in ((1, 0), (67064, 4), (67065, 5)):
            with self.subTest(free_bytes=free_bytes):
                self.assertEqual(sweep.batch_limit(free_bytes, 128, 64, None), expected)

    def test_user_cap_can_only_reduce_the_memory_based_limit(self):
        self.assertEqual(sweep.batch_limit(67065, 128, 64, 3), 3)
        self.assertEqual(sweep.batch_limit(67065, 128, 64, 10), 5)


if __name__ == "__main__":
    unittest.main()
