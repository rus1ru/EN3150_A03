"""EfficientNet-B0 compatibility with the shared EuroSAT data contract."""

import unittest

import torch
from PIL import Image

from en3150_a03.models.efficientnet_b0 import build_model, get_transforms


class EfficientNetB0Tests(unittest.TestCase):
    def test_logits_at_native_resolution(self):
        model = build_model(num_classes=10, pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 10))
        self.assertEqual(model.classifier[1].out_features, 10)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_transforms_do_not_resize(self):
        image = Image.new("RGB", (64, 64), "white")
        for train in (False, True):
            self.assertEqual(tuple(get_transforms(train)(image).shape), (3, 64, 64))

    def test_invalid_class_count(self):
        with self.assertRaises(ValueError):
            build_model(num_classes=1, pretrained=False)


if __name__ == "__main__":
    unittest.main()
