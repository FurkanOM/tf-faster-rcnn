"""Dataset loading and image preprocessing helpers."""

from __future__ import annotations

import os
from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image


DatasetSample = Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor"]
AugmentationFn = Callable[["tf.Tensor", "tf.Tensor"], Tuple["tf.Tensor", "tf.Tensor"]]


def preprocessing(
    image_data: Mapping[str, Any],
    final_height: int,
    final_width: int,
    apply_augmentation: bool = False,
    evaluate: bool = False
) -> DatasetSample:
    """Resize a dataset sample and prepare labels for training or evaluation.

    Args:
        image_data (Mapping[str, Any]): Sample dictionary returned by
            TensorFlow Datasets.
        final_height (int): Output image height after resizing.
        final_width (int): Output image width after resizing.
        apply_augmentation (bool): Whether random horizontal flipping is enabled.
        evaluate (bool): Whether difficult annotations should be filtered out.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Resized image tensor, ground-truth
        boxes, and ground-truth labels.
    """
    img = image_data["image"]
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)
    if evaluate:
        not_diff = tf.logical_not(image_data["objects"]["is_difficult"])
        gt_boxes = gt_boxes[not_diff]
        gt_labels = gt_labels[not_diff]
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if apply_augmentation:
        img, gt_boxes = randomly_apply_operation(flip_horizontally, img, gt_boxes)
    return img, gt_boxes, gt_labels


def get_random_bool() -> tf.Tensor:
    """Return a random boolean tensor for augmentation branching.

    Returns:
        tf.Tensor: Scalar boolean tensor sampled uniformly from `{False, True}`.
    """
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)


def randomly_apply_operation(
    operation: AugmentationFn,
    img: tf.Tensor,
    gt_boxes: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply an augmentation function conditionally to an image and its boxes.

    Args:
        operation (Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]):
            Augmentation function applied to the image and boxes.
        img (tf.Tensor): Image tensor with shape `(height, width, channels)`.
        gt_boxes (tf.Tensor): Bounding boxes with shape `(num_boxes, 4)`.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Possibly transformed image and box tensors.
    """
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes),
        lambda: (img, gt_boxes)
    )


def flip_horizontally(img: tf.Tensor, gt_boxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Flip an image left-to-right and mirror its box coordinates.

    Args:
        img (tf.Tensor): Image tensor with shape `(height, width, channels)`.
        gt_boxes (tf.Tensor): Bounding boxes with shape `(num_boxes, 4)`.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Flipped image and mirrored box coordinates.
    """
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack([gt_boxes[..., 0],
                                1.0 - gt_boxes[..., 3],
                                gt_boxes[..., 2],
                                1.0 - gt_boxes[..., 1]], -1)
    return flipped_img, flipped_gt_boxes


def get_dataset(
    name: str,
    split: str,
    data_dir: str = "~/tensorflow_datasets"
) -> Tuple[tf.data.Dataset, tfds.core.DatasetInfo]:
    """Load a TensorFlow Datasets split together with its metadata.

    Args:
        name (str): Dataset name such as `"voc/2007"` or `"voc/2012"`.
        split (str): Dataset split name.
        data_dir (str): Local dataset cache directory.

    Returns:
        Tuple[tf.data.Dataset, tfds.core.DatasetInfo]: Loaded dataset split and
        associated metadata.
    """
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info


def get_total_item_size(info: tfds.core.DatasetInfo, split: str) -> int:
    """Return the item count for a supported dataset split.

    Args:
        info (tfds.core.DatasetInfo): Dataset metadata object.
        split (str): Dataset split name.

    Returns:
        int: Number of examples in the requested split.
    """
    assert split in ["train", "train+validation", "validation", "test"]
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples


def get_labels(info: tfds.core.DatasetInfo) -> Sequence[str]:
    """Return the dataset label names.

    Args:
        info (tfds.core.DatasetInfo): Dataset metadata object.

    Returns:
        Sequence[str]: Ordered label names exposed by the dataset.
    """
    return info.features["labels"].names


def get_custom_imgs(custom_image_path: str) -> List[str]:
    """Collect image file paths from the top level of the custom image directory.

    Args:
        custom_image_path (str): Directory containing custom images.

    Returns:
        List[str]: Top-level image file paths discovered under the directory.
    """
    img_paths: List[str] = []
    for path, _, filenames in os.walk(custom_image_path):
        for filename in filenames:
            img_paths.append(os.path.join(path, filename))
        break
    return img_paths


def build_custom_dataset(
    img_paths: Sequence[str],
    final_height: int,
    final_width: int
) -> tf.data.Dataset:
    """Build a dataset for local images using the project dataset contract.

    Args:
        img_paths (Sequence[str]): Input image paths.
        final_height (int): Output image height after resizing.
        final_width (int): Output image width after resizing.

    Returns:
        tf.data.Dataset: Dataset yielding resized images and placeholder labels.
    """
    return tf.data.Dataset.from_generator(
        lambda: custom_data_generator(img_paths, final_height, final_width),
        get_data_types(),
        get_data_shapes()
    )


def custom_data_generator(
    img_paths: Sequence[str],
    final_height: int,
    final_width: int
) -> Iterator[DatasetSample]:
    """Yield resized custom images with placeholder annotations.

    Args:
        img_paths (Sequence[str]): Input image paths.
        final_height (int): Output image height after resizing.
        final_width (int): Output image width after resizing.

    Yields:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Resized image tensor, empty
        bounding boxes, and empty labels.
    """
    for img_path in img_paths:
        image = Image.open(img_path)
        resized_image = image.resize((final_width, final_height), Image.LANCZOS)
        img = np.array(resized_image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        yield img, tf.constant([[]], dtype=tf.float32), tf.constant([], dtype=tf.int32)


def get_data_types() -> Tuple[tf.dtypes.DType, tf.dtypes.DType, tf.dtypes.DType]:
    """Return TensorFlow dtypes for image, box, and label tensors.

    Returns:
        Tuple[tf.dtypes.DType, tf.dtypes.DType, tf.dtypes.DType]: Dtypes for the
        image, bounding-box, and label tensors.
    """
    return (tf.float32, tf.float32, tf.int32)


def get_data_shapes() -> Tuple[List[Optional[int]], List[Optional[int]], List[Optional[int]]]:
    """Return dynamic shapes for the batched dataset output.

    Returns:
        Tuple[List[Optional[int]], List[Optional[int]], List[Optional[int]]]:
        Dynamic tensor shapes for image, box, and label batches.
    """
    return ([None, None, None], [None, None], [None,])


def get_padding_values() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return padding values that match the dataset tensor dtypes.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Padding tensors for images, boxes,
        and labels.
    """
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))


def build_dataset(
    dataset: tf.data.Dataset,
    final_height: int,
    final_width: int,
    batch_size: int,
    apply_augmentation: bool = False,
    evaluate: bool = False
) -> tf.data.Dataset:
    """Map preprocessing and padding over a dataset split.

    Args:
        dataset (tf.data.Dataset): Dataset yielding raw TFDS samples.
        final_height (int): Output image height after resizing.
        final_width (int): Output image width after resizing.
        batch_size (int): Number of examples per batch.
        apply_augmentation (bool): Whether random flipping is enabled.
        evaluate (bool): Whether difficult annotations should be filtered out.

    Returns:
        tf.data.Dataset: Preprocessed and padded dataset ready for the model.
    """
    dataset = dataset.map(
        lambda sample: preprocessing(
            sample,
            final_height,
            final_width,
            apply_augmentation=apply_augmentation,
            evaluate=evaluate
        )
    )
    return dataset.padded_batch(
        batch_size,
        padded_shapes=get_data_shapes(),
        padding_values=get_padding_values()
    )
