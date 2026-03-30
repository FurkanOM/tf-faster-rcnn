"""Unit tests for dataset metadata and local file helpers."""

from __future__ import annotations

import os
import tempfile
import types
import unittest

from tests.test_support import import_project_module


class DataUtilsTests(unittest.TestCase):
    """Verify helper functions that do not require TensorFlow datasets at runtime."""

    def setUp(self) -> None:
        pil_module = types.ModuleType("PIL")
        pil_module.Image = types.SimpleNamespace(LANCZOS="LANCZOS")
        numpy_module = types.ModuleType("numpy")
        tfds_module = types.ModuleType("tensorflow_datasets")
        self.module, _ = import_project_module(
            "utils.data_utils",
            extra_modules={
                "PIL": pil_module,
                "numpy": numpy_module,
                "tensorflow_datasets": tfds_module,
            },
            clear_modules=("utils.data_utils",),
        )

    def test_get_total_item_size_supports_combined_split(self) -> None:
        info = types.SimpleNamespace(
            splits={
                "train": types.SimpleNamespace(num_examples=8),
                "validation": types.SimpleNamespace(num_examples=3),
                "test": types.SimpleNamespace(num_examples=5),
            }
        )

        self.assertEqual(self.module.get_total_item_size(info, "train+validation"), 11)
        self.assertEqual(self.module.get_total_item_size(info, "test"), 5)

    def test_get_labels_returns_dataset_label_names(self) -> None:
        info = types.SimpleNamespace(
            features={"labels": types.SimpleNamespace(names=["person", "car", "dog"])}
        )

        self.assertEqual(self.module.get_labels(info), ["person", "car", "dog"])

    def test_get_custom_imgs_returns_top_level_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            top_level_files = [
                os.path.join(tempdir, "first.jpg"),
                os.path.join(tempdir, "second.png"),
            ]
            nested_dir = os.path.join(tempdir, "nested")
            os.makedirs(nested_dir)
            nested_file = os.path.join(nested_dir, "ignored.jpg")

            for path in top_level_files + [nested_file]:
                with open(path, "w", encoding="utf-8") as file_obj:
                    file_obj.write("placeholder")

            img_paths = self.module.get_custom_imgs(tempdir)

        self.assertCountEqual(img_paths, top_level_files)


if __name__ == "__main__":
    unittest.main()
