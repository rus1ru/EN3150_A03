"""Tests for Model A experiment classification metrics."""

import unittest

from en3150_a03.model_a_metrics import classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_accuracy_macro_scores_and_confusion_matrix(self):
        result = classification_metrics(
            targets=[0, 0, 1, 1, 2],
            predictions=[0, 1, 1, 1, 0],
            class_names=["A", "B", "C"],
        )
        self.assertAlmostEqual(result["accuracy"], 3 / 5)
        self.assertAlmostEqual(result["macro_precision"], (0.5 + 2 / 3 + 0) / 3)
        self.assertAlmostEqual(result["macro_recall"], (0.5 + 1 + 0) / 3)
        self.assertEqual(
            result["confusion_matrix"],
            [[1, 1, 0], [0, 2, 0], [1, 0, 0]],
        )
        self.assertEqual(result["per_class"]["C"]["support"], 1)

    def test_rejects_mismatched_or_out_of_range_labels(self):
        with self.assertRaises(ValueError):
            classification_metrics([0], [], ["A", "B"])
        with self.assertRaises(ValueError):
            classification_metrics([2], [0], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
