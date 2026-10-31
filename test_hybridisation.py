"""Method checks using only synthetic observations and standard-library unittest."""

import random
import unittest

import pandas as pd

from Data_Hybridisation_fn import Data_Hybridisation
from example_synthetic import make_source


class HybridisationTests(unittest.TestCase):
    def setUp(self):
        self.first = make_source("source_1", 0.0)
        self.second = make_source("source_2", 10.0)

    def test_sampling_preserves_whole_ordered_windows_and_inputs(self):
        # Reverse input rows to check chronological ordering within each block.
        first = self.first.iloc[::-1].copy()
        second = self.second.loc[:, ["source", "value", "timestamp"]].copy()
        before_first, before_second = first.copy(), second.copy()
        result = Data_Hybridisation(first, second, 0.25, random_state=42)
        self.assertEqual(result["source"].value_counts().to_dict(),
                         {"source_1": 45, "source_2": 15})
        self.assertEqual(list(result.columns), list(first.columns))
        self.assertFalse(result.duplicated(["source", "timestamp"]).any())
        for start in range(0, len(result), 5):
            block = result.iloc[start:start + 5]
            self.assertEqual(block["source"].nunique(), 1)
            self.assertEqual(block["timestamp"].dt.floor("5min").nunique(), 1)
            self.assertTrue(block["timestamp"].is_monotonic_increasing)
            self.assertEqual(block["timestamp"].dt.minute.iloc[0] % 5, 0)
        pd.testing.assert_frame_equal(first, before_first)
        pd.testing.assert_frame_equal(second, before_second)

    def test_seed_is_repeatable_and_does_not_change_global_random_state(self):
        before = random.getstate()
        a = Data_Hybridisation(self.first, self.second, random_state=42)
        b = Data_Hybridisation(self.first, self.second, random_state=42)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(before, random.getstate())

    def test_endpoints_rounding_and_shortage(self):
        for ratio, expected in [(0, self.first), (1, self.second.iloc[:10])]:
            result = Data_Hybridisation(self.first, self.second.iloc[:10], ratio,
                                       random_state=42)
            pd.testing.assert_frame_equal(
                result.sort_values("timestamp").reset_index(drop=True),
                expected.reset_index(drop=True),
            )
        rounded = Data_Hybridisation(self.first, self.second, 0.01, random_state=42)
        self.assertEqual(set(rounded["source"]), {"source_1"})
        with self.assertWarns(RuntimeWarning):
            limited = Data_Hybridisation(self.first, self.second.iloc[:10], 0.5,
                                        random_state=42)
        self.assertEqual(limited["source"].value_counts().to_dict(),
                         {"source_1": 50, "source_2": 10})

    def test_partial_windows_gaps_and_existing_metadata_column(self):
        sparse = self.first.iloc[[0, 1, 5, 20]].copy()
        sparse["_hybridisation_window"] = ["a", "b", "c", "d"]
        other = sparse.copy()
        other["source"] = "source_2"
        result = Data_Hybridisation(sparse, other, 0, random_state=42)
        pd.testing.assert_frame_equal(
            result.sort_values("timestamp").reset_index(drop=True),
            sparse.reset_index(drop=True),
        )

    def test_invalid_inputs(self):
        for ratio in [-0.1, 1.1, float("nan")]:
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                Data_Hybridisation(self.first, self.second, ratio)
        for duration in ["0min", "-5min", "invalid"]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                Data_Hybridisation(self.first, self.second, window_length=duration)
        with self.assertRaises(TypeError):
            Data_Hybridisation(self.first, self.second, True)
        with self.assertRaises(ValueError):
            Data_Hybridisation(self.first.iloc[:0], self.second)
        with self.assertRaises(ValueError):
            Data_Hybridisation(self.first.drop(columns="value"), self.second)
        invalid = self.first.copy()
        invalid.loc[0, "timestamp"] = pd.NaT
        with self.assertRaises(ValueError):
            Data_Hybridisation(invalid, self.second)
        duplicate = self.first.copy()
        duplicate.columns = ["timestamp", "value", "value"]
        with self.assertRaises(ValueError):
            Data_Hybridisation(duplicate, duplicate)


if __name__ == "__main__":
    unittest.main()
