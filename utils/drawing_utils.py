"""Visualization helpers for proposals and final detections."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageDraw

from utils import bbox_utils


def draw_grid_map(img: tf.Tensor, grid_map: tf.Tensor, stride: int) -> None:
    """Render anchor-grid intersection points over an image.

    Args:
        img (tf.Tensor): Image tensor with shape `(height, width, channels)`.
        grid_map (tf.Tensor): Grid coordinates with shape `(total_points, 4)`.
        stride (int): Feature-map stride measured in pixels.

    Returns:
        None: The rendered image is displayed with Matplotlib.
    """
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    for grid in grid_map:
        draw.rectangle((
            grid[0] + stride // 2 - 2,
            grid[1] + stride // 2 - 2,
            grid[2] + stride // 2 + 2,
            grid[3] + stride // 2 + 2), fill=(255, 255, 255, 0))
    plt.figure()
    plt.imshow(image)
    plt.show()


def draw_bboxes(imgs: tf.Tensor, bboxes: tf.Tensor) -> None:
    """Draw normalized bounding boxes on a batch of images.

    Args:
        imgs (tf.Tensor): Image batch with shape
            `(batch_size, height, width, channels)`.
        bboxes (tf.Tensor): Normalized boxes with shape
            `(batch_size, total_bboxes, 4)`.

    Returns:
        None: Each image is displayed with Matplotlib.
    """
    colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)
    plt.figure()
    for img_with_bb in imgs_with_bb:
        plt.imshow(img_with_bb)
        plt.show()


def draw_bboxes_with_labels(
    img: tf.Tensor,
    bboxes: tf.Tensor,
    label_indices: tf.Tensor,
    probs: tf.Tensor,
    labels: Sequence[str]
) -> None:
    """Draw denormalized boxes and labels on a single image.

    Args:
        img (tf.Tensor): Image tensor with shape `(height, width, channels)`.
        bboxes (tf.Tensor): Pixel-space boxes with shape `(total_bboxes, 4)`.
        label_indices (tf.Tensor): Label indices with shape `(total_bboxes,)`.
        probs (tf.Tensor): Confidence scores with shape `(total_bboxes,)`.
        labels (Sequence[str]): Ordered label names including background.

    Returns:
        None: The rendered image is displayed with Matplotlib.
    """
    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
    image = tf.keras.preprocessing.image.array_to_img(img)
    draw = ImageDraw.Draw(image)
    for index, bbox in enumerate(bboxes):
        y1, x1, y2, x2 = tf.split(bbox, 4)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        label_index = int(label_indices[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], probs[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    plt.figure()
    plt.imshow(image)
    plt.show()


def draw_predictions(
    dataset: tf.data.Dataset,
    pred_bboxes: tf.Tensor,
    pred_labels: tf.Tensor,
    pred_scores: tf.Tensor,
    labels: Sequence[str],
    batch_size: int
) -> None:
    """Iterate over predictions and display the rendered detections.

    Args:
        dataset (tf.data.Dataset): Dataset yielding batches of input images.
        pred_bboxes (tf.Tensor): Predicted normalized boxes for the full dataset.
        pred_labels (tf.Tensor): Predicted label indices for the full dataset.
        pred_scores (tf.Tensor): Predicted confidence scores for the full dataset.
        labels (Sequence[str]): Ordered label names including background.
        batch_size (int): Batch size used during prediction.

    Returns:
        None: Each image is displayed with its predicted detections.
    """
    for batch_id, image_data in enumerate(dataset):
        imgs, _, _ = image_data
        img_size = imgs.shape[1]
        start = batch_id * batch_size
        end = start + batch_size
        batch_bboxes = pred_bboxes[start:end]
        batch_labels = pred_labels[start:end]
        batch_scores = pred_scores[start:end]
        for i, img in enumerate(imgs):
            denormalized_bboxes = bbox_utils.denormalize_bboxes(batch_bboxes[i], img_size, img_size)
            draw_bboxes_with_labels(img, denormalized_bboxes, batch_labels[i], batch_scores[i], labels)
