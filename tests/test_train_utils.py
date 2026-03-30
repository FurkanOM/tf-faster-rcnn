"""Unit tests for training helper utilities."""

from __future__ import annotations

import copy
import unittest

from tests.test_support import import_project_module


class TrainUtilsTests(unittest.TestCase):
    """Cover hyper-parameter and batching helpers."""

    def setUp(self) -> None:
        self.module, _ = import_project_module(
            "utils.train_utils",
            clear_modules=("utils.train_utils", "utils.bbox_utils"),
        )
        original_rpn = copy.deepcopy(self.module.RPN)

        def restore_rpn() -> None:
            self.module.RPN.clear()
            self.module.RPN.update(copy.deepcopy(original_rpn))

        self.addCleanup(restore_rpn)

    def test_get_hyper_params_populates_defaults_and_anchor_count(self) -> None:
        params = self.module.get_hyper_params("vgg16")

        self.assertEqual(params["img_size"], 500)
        self.assertEqual(params["pre_nms_topn"], 6000)
        self.assertEqual(params["train_nms_topn"], 1500)
        self.assertEqual(params["test_nms_topn"], 300)
        self.assertEqual(params["anchor_count"], 9)

    def test_get_hyper_params_applies_truthy_overrides(self) -> None:
        params = self.module.get_hyper_params(
            "mobilenet_v2",
            img_size=320,
            pre_nms_topn=1024,
            total_pos_bboxes=64,
        )

        self.assertEqual(params["img_size"], 320)
        self.assertEqual(params["pre_nms_topn"], 1024)
        self.assertEqual(params["total_pos_bboxes"], 64)
        self.assertEqual(params["anchor_count"], 9)

    def test_get_hyper_params_returns_a_copy_of_backbone_defaults(self) -> None:
        params = self.module.get_hyper_params("vgg16")

        params["img_size"] = 320

        self.assertEqual(self.module.RPN["vgg16"]["img_size"], 500)

    def test_get_step_size_rounds_up_for_partial_batches(self) -> None:
        self.assertEqual(self.module.get_step_size(10, 4), 3)
        self.assertEqual(self.module.get_step_size(8, 4), 2)


if __name__ == "__main__":
    unittest.main()
