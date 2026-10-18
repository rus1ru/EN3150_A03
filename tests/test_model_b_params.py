"""Model B parameter budget, convolution grouping, shapes and compute checks."""
import unittest
import torch
from torch import nn
from en3150_a03.models.model_b import ModelB
from en3150_a03.models.model_a import ModelA
from en3150_a03.model_b_measure import profile_model


class ModelBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shape_backward_grouping_and_budget(self):
        model = ModelB()
        x = torch.randn(3, 3, 64, 64)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (3, 10))
        nn.functional.cross_entropy(logits, torch.tensor([0, 1, 9])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5589)
        self.assertLessEqual(sum(p.numel() for p in model.parameters()), 100_000)
        convolutions = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        self.assertEqual(len(convolutions), 6)
        for dw, pw in zip(convolutions[::2], convolutions[1::2]):
            self.assertEqual(dw.groups, dw.in_channels)
            self.assertEqual(dw.out_channels, dw.in_channels)
            self.assertEqual(dw.kernel_size, (3, 3))
            self.assertEqual(pw.kernel_size, (1, 1))

    def test_hand_calculated_macs_and_profile_does_not_change_model(self):
        model = ModelB()
        model.train()
        profile = profile_model(model)
        self.assertTrue(model.training)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))
        # H*W*(9*Cin + Cin*Cout), summed over the three blocks + dense.
        expected = 64**2*(9*3+3*16) + 32**2*(9*16+16*32) + 16**2*(9*32+32*64) + 64*32+32*10
        self.assertEqual(profile["macs"], expected)
        self.assertEqual(expected, 1_579_328)
        self.assertEqual(profile_model(ModelA())["macs"], 11_209_024)
        self.assertEqual(sum(row["parameters"] for row in profile["layers"]), 5589)
        self.assertAlmostEqual(profile["estimated_float32_size_kb"], 21.83203125)

    def test_class_validation(self):
        self.assertEqual(ModelB(2)(torch.rand(1, 3, 64, 64)).shape, (1, 2))
        with self.assertRaises(ValueError):
            ModelB(1)
        with self.assertRaises(ValueError):
            ModelB(4000)

if __name__ == "__main__":
    unittest.main()
