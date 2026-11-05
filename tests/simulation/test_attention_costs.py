"""Hand-computed attention fixtures check interpolation, not GPU accuracy."""

import csv
import math
from pathlib import Path
import tempfile
import unittest

from simulation.costs import AttentionCostModel


class AttentionCostTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def profile(self, rows, *, components=False):
        path = self.root / "attention.csv"
        fields = ["batch_size", "context_length", "time_ms"]
        if components:
            fields.append("component")
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
        return path

    def test_nonuniform_grid_interpolates_components_and_converts_units_once(self):
        rows = []
        for batch, context, core in ((1, 100, 1), (1, 500, 3),
                                     (5, 100, 5), (5, 500, 11)):
            rows.extend(((batch, context, core, "attention"),
                         (batch, context, batch + 1, "qkv_proj"),
                         (batch, context, 0.25, "norm"),
                         (batch, context, 0, "rope")))
        cost = AttentionCostModel(999, self.profile(reversed(rows), components=True),
                                  ticks_per_millisecond=7)
        estimate = cost.evaluate([100, 300])
        # B=2 and L=200 each lie one quarter of the way across their axes.
        # Core=2.75 ms, projection=3 ms, norm=.25 ms: exactly 6 ms total.
        self.assertEqual(estimate.batch_size, 2)
        self.assertEqual(estimate.context_mean, 200)
        self.assertEqual(estimate.components_ticks,
                         {"attention": 19.25, "qkv_proj": 21, "norm": 1.75, "rope": 0})
        self.assertEqual(estimate.total_ticks, 42)
        self.assertEqual(cost([100, 300]), estimate.total_ticks)
        self.assertEqual(cost([100]), 22.75)
        self.assertEqual(cost([500] * 5), 120.75)
        description = cost.describe()
        self.assertEqual(description["context_aggregation"], "arithmetic_mean")
        self.assertEqual(description["interpolation"], "bilinear")
        self.assertEqual(description["out_of_range"], "error")
        self.assertEqual(description["ticks_per_millisecond"], 7)

    def test_single_axis_interpolation_retains_fractional_context_mean(self):
        cost = AttentionCostModel(999, self.profile(((2, 100, 1), (2, 102, 2))),
                                  ticks_per_millisecond=8)
        estimate = cost.evaluate([100, 101])
        self.assertEqual(estimate.context_mean, 100.5)
        self.assertEqual(estimate.components_ticks, {"total": 10})
        self.assertEqual(cost([100, 100]), 8)
        self.assertEqual(cost([102, 102]), 16)

    def test_single_context_axis_interpolates_batch_without_padding(self):
        cost = AttentionCostModel(999, self.profile(((1, 0, 1), (4, 0, 7))))
        self.assertEqual(cost([0, 0]), 30)
        self.assertEqual(cost([0, 0, 0]), 50)

    def test_empty_batch_and_unprofiled_fallback_remain_in_ticks(self):
        cost = AttentionCostModel(0.25, ticks_per_millisecond=1000)
        self.assertEqual(cost([0, 100000]), 0.5)
        self.assertEqual(cost.evaluate([0, 100000]).components_ticks,
                         {"linear_per_token_proxy": 0.5})
        for model in (cost, AttentionCostModel(1, self.profile(((1, 10, 0.1),)))):
            with self.subTest(profile=model.profile_path):
                empty = model.evaluate([])
                self.assertEqual(empty.batch_size, 0)
                self.assertIsNone(empty.context_mean)
                self.assertEqual(empty.total_ticks, 0)
                self.assertTrue(all(value == 0 for value in empty.components_ticks.values()))
        self.assertEqual(AttentionCostModel(0)([0]), 0)

    def test_queries_outside_either_axis_fail_without_fallback(self):
        cost = AttentionCostModel(123, self.profile(((2, 100, 1), (2, 300, 2),
                                                   (4, 100, 3), (4, 300, 4))))
        for contexts in ([200], [200] * 5, [99, 99], [301, 301]):
            with self.subTest(contexts=contexts), self.assertRaisesRegex(ValueError, "outside measured range"):
                cost(contexts)

    def test_missing_corner_fails_but_does_not_invalidate_exact_measurements(self):
        cost = AttentionCostModel(123, self.profile(((1, 100, 1), (1, 300, 2),
                                                   (3, 100, 3))))
        self.assertEqual(cost([300]), 20)
        with self.assertRaisesRegex(ValueError, "missing interpolation corners"):
            cost([200, 200])

    def test_components_must_be_complete_disjoint_and_unique(self):
        cases = (
            (((1, 10, 1, "attention"), (1, 10, 1, "attention")), "Duplicate"),
            (((1, 10, 1, "attention"), (1, 10, 1, "qkv_proj"),
              (1, 20, 1, "attention")), "missing components"),
            (((1, 10, 2, "total"), (1, 10, 1, "attention")), "cannot mix total"),
            (((1, 10, 1, "total"), (1, 20, 1, "attention")), "cannot mix total"),
            (((1, 10, 1, " "),), "Invalid attention"),
        )
        for rows, message in cases:
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, message):
                AttentionCostModel(1, self.profile(rows, components=True))

    def test_invalid_profile_axes_and_times_fail_at_load(self):
        for row in ((0, 10, 1), (-1, 10, 1), (1.5, 10, 1), (1, -1, 1),
                    (1, 1.5, 1), (1, 10, -1), (1, 10, math.nan),
                    (1, 10, math.inf), (1, 10, "bad"), (1, 10, 0)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                AttentionCostModel(1, self.profile((row,)))

    def test_invalid_query_counts_and_units_fail(self):
        for contexts in ([-1], [1.5], [True], [math.nan], "12", None):
            with self.subTest(contexts=contexts), self.assertRaises(ValueError):
                AttentionCostModel(1).evaluate(contexts)
        for units in (0, -1, math.nan, math.inf):
            with self.subTest(units=units), self.assertRaises(ValueError):
                AttentionCostModel(1, ticks_per_millisecond=units)
        for fallback in (-1, math.nan, math.inf):
            with self.subTest(fallback=fallback), self.assertRaises(ValueError):
                AttentionCostModel(fallback)

    def test_invalid_csv_schema_and_missing_component_cell_fail(self):
        path = self.root / "malformed.csv"
        for contents in (
            "batch_size,context_length,time_ms\n",
            "batch_size,context_length\n1,10\n",
            "batch_size,context_length,time_ms,time_ms\n1,10,1,2\n",
            "batch_size,context_length,time_ms,component\n1,10,1\n",
        ):
            path.write_text(contents)
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                AttentionCostModel(1, path)


if __name__ == "__main__":
    unittest.main()
