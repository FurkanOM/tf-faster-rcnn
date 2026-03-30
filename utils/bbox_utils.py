"""Bounding-box helpers used by the RPN and Faster R-CNN heads."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import tensorflow as tf


HyperParams = Dict[str, Any]


def generate_base_anchors(hyper_params: HyperParams) -> tf.Tensor:
    """Create anchor templates centered at the origin for each ratio and scale.

    Args:
        hyper_params (Dict[str, Any]): Hyper-parameter dictionary containing image
            size, anchor ratios, and anchor scales.

    Returns:
        tf.Tensor: Base anchors with shape `(anchor_count, 4)` in normalized
        coordinates relative to the image size.
    """
    img_size = hyper_params["img_size"]
    anchor_ratios = hyper_params["anchor_ratios"]
    anchor_scales = hyper_params["anchor_scales"]
    base_anchors = []
    for scale in anchor_scales:
        scale /= img_size
        for ratio in anchor_ratios:
            w = tf.sqrt(scale ** 2 / ratio)
            h = w * ratio
            base_anchors.append([-h / 2, -w / 2, h / 2, w / 2])
    return tf.cast(base_anchors, dtype=tf.float32)


def generate_anchors(hyper_params: HyperParams) -> tf.Tensor:
    """Broadcast anchor templates over the feature-map grid in normalized coordinates.

    Args:
        hyper_params (Dict[str, Any]): Hyper-parameter dictionary containing the
            feature-map size and anchor configuration.

    Returns:
        tf.Tensor: Anchors with shape `(total_anchors, 4)` clipped to `[0, 1]`.
    """
    anchor_count = hyper_params["anchor_count"]
    feature_map_shape = hyper_params["feature_map_shape"]
    stride = 1 / feature_map_shape
    grid_coords = tf.cast(tf.range(0, feature_map_shape) / feature_map_shape + stride / 2, dtype=tf.float32)
    grid_x, grid_y = tf.meshgrid(grid_coords, grid_coords)
    flat_grid_x = tf.reshape(grid_x, (-1,))
    flat_grid_y = tf.reshape(grid_y, (-1,))
    grid_map = tf.stack([flat_grid_y, flat_grid_x, flat_grid_y, flat_grid_x], axis=-1)
    base_anchors = generate_base_anchors(hyper_params)
    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))
    anchors = tf.reshape(anchors, (-1, 4))
    return tf.clip_by_value(anchors, 0, 1)


def non_max_suppression(
    pred_bboxes: tf.Tensor,
    pred_labels: tf.Tensor,
    **kwargs: Any
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Delegate batched non-maximum suppression to TensorFlow.

    Args:
        pred_bboxes (tf.Tensor): Candidate boxes with shape
            `(batch_size, total_bboxes, total_labels, 4)`.
        pred_labels (tf.Tensor): Class scores with shape
            `(batch_size, total_bboxes, total_labels)`.
        **kwargs (Any): Keyword arguments forwarded to
            `tf.image.combined_non_max_suppression`.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Selected boxes, scores,
        classes, and valid detection counts.
    """
    return tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        **kwargs
    )


def get_bboxes_from_deltas(anchors: tf.Tensor, deltas: tf.Tensor) -> tf.Tensor:
    """Decode box deltas back into bounding boxes.

    Args:
        anchors (tf.Tensor): Anchor boxes with shape `(..., 4)`.
        deltas (tf.Tensor): Encoded box deltas with shape `(..., 4)`.

    Returns:
        tf.Tensor: Decoded bounding boxes with shape `(..., 4)`.
    """
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[..., 0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height
    all_bbox_width = tf.exp(deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_anc_height) + all_anc_ctr_y
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    return tf.stack([y1, x1, y2, x2], axis=-1)


def get_deltas_from_bboxes(bboxes: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """Encode ground-truth boxes relative to proposal boxes.

    Args:
        bboxes (tf.Tensor): Proposal boxes with shape `(..., 4)`.
        gt_boxes (tf.Tensor): Ground-truth boxes with shape `(..., 4)`.

    Returns:
        tf.Tensor: Encoded deltas with shape `(..., 4)`.
    """
    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height
    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height
    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)


def generate_iou_map(bboxes: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """Compute the pairwise IoU map between proposals and ground-truth boxes.

    Args:
        bboxes (tf.Tensor): Proposal boxes with shape
            `(batch_size, total_bboxes, 4)`.
        gt_boxes (tf.Tensor): Ground-truth boxes with shape
            `(batch_size, total_gt_boxes, 4)`.

    Returns:
        tf.Tensor: IoU map with shape `(batch_size, total_bboxes, total_gt_boxes)`.
    """
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    return intersection_area / union_area


def normalize_bboxes(bboxes: tf.Tensor, height: tf.Tensor, width: tf.Tensor) -> tf.Tensor:
    """Convert absolute pixel coordinates into normalized box coordinates.

    Args:
        bboxes (tf.Tensor): Bounding boxes with shape `(..., 4)` in pixels.
        height (tf.Tensor): Image height in pixels.
        width (tf.Tensor): Image width in pixels.

    Returns:
        tf.Tensor: Bounding boxes with shape `(..., 4)` normalized to `[0, 1]`.
    """
    y1 = bboxes[..., 0] / height
    x1 = bboxes[..., 1] / width
    y2 = bboxes[..., 2] / height
    x2 = bboxes[..., 3] / width
    return tf.stack([y1, x1, y2, x2], axis=-1)


def denormalize_bboxes(bboxes: tf.Tensor, height: tf.Tensor, width: tf.Tensor) -> tf.Tensor:
    """Convert normalized box coordinates back into pixel coordinates.

    Args:
        bboxes (tf.Tensor): Bounding boxes with shape `(..., 4)` normalized to
            `[0, 1]`.
        height (tf.Tensor): Image height in pixels.
        width (tf.Tensor): Image width in pixels.

    Returns:
        tf.Tensor: Bounding boxes with shape `(..., 4)` in pixel coordinates.
    """
    y1 = bboxes[..., 0] * height
    x1 = bboxes[..., 1] * width
    y2 = bboxes[..., 2] * height
    x2 = bboxes[..., 3] * width
    return tf.round(tf.stack([y1, x1, y2, x2], axis=-1))
