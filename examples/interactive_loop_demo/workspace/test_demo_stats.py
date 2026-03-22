from __future__ import annotations

import unittest

from demo_stats import moving_average, normalize_scores


class DemoStatsTests(unittest.TestCase):
    def test_moving_average_slides_over_the_window(self) -> None:
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])

    def test_moving_average_collapses_large_windows(self) -> None:
        self.assertEqual(moving_average([2, 4, 6], 5), [4.0])

    def test_normalize_scores_scales_against_the_maximum(self) -> None:
        self.assertEqual(normalize_scores([2, 4, 8]), [0.25, 0.5, 1.0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
