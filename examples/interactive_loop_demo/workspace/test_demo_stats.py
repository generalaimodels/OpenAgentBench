"""Unit tests for the compatibility demo workspace."""

from __future__ import annotations

import unittest

from demo_stats import mean, minimum


class DemoStatsTests(unittest.TestCase):
    def test_mean(self) -> None:
        self.assertAlmostEqual(mean([1.0, 2.0, 3.0]), 2.0)

    def test_minimum(self) -> None:
        self.assertEqual(minimum([4.0, 2.0, 9.0]), 2.0)

    def test_empty_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            mean([])


if __name__ == "__main__":
    unittest.main()
