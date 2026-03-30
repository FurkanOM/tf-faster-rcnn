"""Unit tests for filesystem and argument helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from unittest import mock

from tests.test_support import import_project_module, make_tf_stub


class IoUtilsTests(unittest.TestCase):
    """Exercise I/O helpers without requiring TensorFlow to be installed."""

    def test_get_log_path_includes_timestamp(self) -> None:
        module, _ = import_project_module("utils.io_utils")

        class FixedDateTime:
            @staticmethod
            def now() -> datetime:
                return datetime(2020, 1, 2, 3, 4, 5)

        with mock.patch.object(module, "datetime", FixedDateTime):
            log_path = module.get_log_path("rpn", backbone="mobilenet_v2", custom_postfix="_debug")

        self.assertEqual(log_path, "logs/rpn_mobilenet_v2_debug/20200102-030405")

    def test_get_model_path_creates_trained_directory(self) -> None:
        module, _ = import_project_module("utils.io_utils")

        with tempfile.TemporaryDirectory() as tempdir:
            current_dir = os.getcwd()
            self.addCleanup(os.chdir, current_dir)
            os.chdir(tempdir)

            model_path = module.get_model_path("faster_rcnn", backbone="vgg16")

            self.assertTrue(os.path.isdir(os.path.join(tempdir, "trained")))
            self.assertEqual(model_path, "trained/faster_rcnn_vgg16_model_weights.h5")

    def test_handle_args_parses_gpu_flag_and_backbone(self) -> None:
        module, _ = import_project_module("utils.io_utils")

        with mock.patch("sys.argv", ["prog", "-handle-gpu", "--backbone", "vgg16"]):
            args = module.handle_args()

        self.assertTrue(args.handle_gpu)
        self.assertEqual(args.backbone, "vgg16")

    def test_is_valid_backbone_rejects_unknown_values(self) -> None:
        module, _ = import_project_module("utils.io_utils")

        module.is_valid_backbone("mobilenet_v2")
        with self.assertRaises(AssertionError):
            module.is_valid_backbone("resnet50")

    def test_handle_gpu_compatibility_enables_memory_growth_for_each_gpu(self) -> None:
        tf_stub = make_tf_stub(gpus=["GPU:0", "GPU:1"])
        module, _ = import_project_module("utils.io_utils", tensorflow_stub=tf_stub)

        module.handle_gpu_compatibility()

        self.assertEqual(
            tf_stub.memory_growth_calls,
            [("GPU:0", True), ("GPU:1", True)],
        )


if __name__ == "__main__":
    unittest.main()
