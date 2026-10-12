"""Model A shape and parameter-count tests."""

import unittest

import torch

from en3150_a03.models.model_a import (
    ModelA,
    count_trainable_parameters,
    estimate_float32_size_kb,
)


class ModelATests(unittest.TestCase):
    def test_forward_shape_for_eurosat_rgb(self):
        model = ModelA(num_classes=10)
        output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 10))

    def test_architecture_parameter_count_and_float32_size(self):
        model = ModelA(num_classes=10)
        self.assertEqual(count_trainable_parameters(model), 25_994)
        self.assertAlmostEqual(estimate_float32_size_kb(model), 101.5390625)


if __name__ == "__main__":
    unittest.main()
