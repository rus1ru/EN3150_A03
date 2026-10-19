"""Tests for explicit Model A device selection."""

import unittest
from unittest.mock import patch

import torch

from en3150_a03.model_a_device import resolve_device


class ModelADeviceTests(unittest.TestCase):
    def test_cpu_can_be_selected_explicitly(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_cpu_only_torch_build_gives_actionable_cuda_error(self):
        with (
            patch.object(torch, "version") as version,
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            version.cuda = None
            with self.assertRaisesRegex(RuntimeError, "is CPU-only"):
                resolve_device("cuda:0")


if __name__ == "__main__":
    unittest.main()
