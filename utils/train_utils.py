"""Training-time utilities shared by the RPN and Faster R-CNN pipelines."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, Tuple

import tensorflow as tf

from utils import bbox_utils


HyperParams = Dict[str, Any]
FasterRCNNInputs = Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor"]
RPNOutputs = Tuple["tf.Tensor", "tf.Tensor"]

RPN: Dict[str, HyperParams] = {
    "vgg16": {
        "img_size": 500,
        "feature_map_shape": 31,
        "anchor_ratios": [1., 2., 1./2.],
        "anchor_scales": [128, 256, 512],
    },
    "mobilenet_v2": {
        "img_size": 500,
        "feature_map_shape": 32,
        "anchor_ratios": [1., 2., 1./2.],
        "anchor_scales": [128, 256, 512],
    }
}


def get_hyper_params(backbone: str, **kwargs: Any) -> HyperParams:
    """Build the hyper-parameter dictionary for the selected backbone.

    Args:
        backbone (str): Backbone name, typically `"vgg16"` or `"mobilenet_v2"`.
        **kwargs (Any): Optional hyper-parameter overrides.

    Returns:
        Dict[str, Any]: Hyper-parameter dictionary used by the training code.
    """
    hyper_params = dict(RPN[backbone])
    hyper_params["pre_nms_topn"] = 6000
    hyper_params["train_nms_topn"] = 1500
    hyper_params["test_nms_topn"] = 300
    hyper_params["nms_iou_threshold"] = 0.7
    hyper_params["total_pos_bboxes"] = 128
    hyper_params["total_neg_bboxes"] = 128
    hyper_params["pooling_size"] = (7, 7)
    hyper_params["variances"] = [0.1, 0.1, 0.2, 0.2]
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value
    hyper_params["anchor_count"] = len(hyper_params["anchor_ratios"]) * len(hyper_params["anchor_scales"])
    return hyper_params


def get_step_size(total_items: int, batch_size: int) -> int:
    """Return the number of steps required to cover a dataset once.

    Args:
        total_items (int): Number of examples in the dataset.
        batch_size (int): Number of examples per batch.

    Returns:
        int: Number of steps required to iterate over the dataset once.
    """
    return math.ceil(total_items / batch_size)


def randomly_select_xyz_mask(mask: tf.Tensor, select_xyz: tf.Tensor) -> tf.Tensor:
    """Keep up to the requested number of true elements per batch row.

    Args:
        mask (tf.Tensor): Boolean mask with shape `(batch_size, total_items)`.
        select_xyz (tf.Tensor): Number of `True` entries to keep per batch row.

    Returns:
        tf.Tensor: Boolean mask with the same shape as `mask`.
    """
    maxval = tf.reduce_max(select_xyz) * 10
    random_mask = tf.random.uniform(tf.shape(mask), minval=1, maxval=maxval, dtype=tf.int32)
    multiplied_mask = tf.cast(mask, tf.int32) * random_mask
    sorted_mask = tf.argsort(multiplied_mask, direction="DESCENDING")
    sorted_mask_indices = tf.argsort(sorted_mask)
    selected_mask = tf.less(sorted_mask_indices, tf.expand_dims(select_xyz, 1))
    return tf.logical_and(mask, selected_mask)


def faster_rcnn_generator(
    dataset: tf.data.Dataset,
    anchors: tf.Tensor,
    hyper_params: HyperParams
) -> Iterator[Tuple[FasterRCNNInputs, Tuple[()]]]:
    """Yield batched Faster R-CNN inputs and placeholder outputs for fitting.

    Args:
        dataset (tf.data.Dataset): Padded dataset yielding image, box, and label
            tensors.
        anchors (tf.Tensor): Anchor tensor with shape `(total_anchors, 4)`.
        hyper_params (Dict[str, Any]): Hyper-parameter dictionary.

    Yields:
        Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], Tuple[()]]:
        Batched Faster R-CNN inputs and the empty target tuple expected by Keras.
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            bbox_deltas, bbox_labels = _get_rpn_training_targets(gt_boxes, gt_labels, anchors, hyper_params)
            yield (img, gt_boxes, gt_labels, bbox_deltas, bbox_labels), ()


def rpn_generator(
    dataset: tf.data.Dataset,
    anchors: tf.Tensor,
    hyper_params: HyperParams
) -> Iterator[Tuple[tf.Tensor, RPNOutputs]]:
    """Yield batched RPN inputs and supervision targets for fitting.

    Args:
        dataset (tf.data.Dataset): Padded dataset yielding image, box, and label
            tensors.
        anchors (tf.Tensor): Anchor tensor with shape `(total_anchors, 4)`.
        hyper_params (Dict[str, Any]): Hyper-parameter dictionary.

    Yields:
        Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]: Input images together with
        regression and classification targets.
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            bbox_deltas, bbox_labels = _get_rpn_training_targets(gt_boxes, gt_labels, anchors, hyper_params)
            yield img, (bbox_deltas, bbox_labels)


def _get_rpn_training_targets(
    gt_boxes: tf.Tensor,
    gt_labels: tf.Tensor,
    anchors: tf.Tensor,
    hyper_params: HyperParams
) -> RPNOutputs:
    """Build RPN targets for a single batched dataset element."""
    return calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params)


def calculate_rpn_actual_outputs(
    anchors: tf.Tensor,
    gt_boxes: tf.Tensor,
    gt_labels: tf.Tensor,
    hyper_params: HyperParams
) -> RPNOutputs:
    """Create RPN regression and classification targets for a batch.

    Args:
        anchors (tf.Tensor): Anchor tensor with shape `(total_anchors, 4)`.
        gt_boxes (tf.Tensor): Ground-truth boxes with shape
            `(batch_size, padded_gt_boxes, 4)`.
        gt_labels (tf.Tensor): Ground-truth labels with shape
            `(batch_size, padded_gt_boxes)`.
        hyper_params (Dict[str, Any]): Hyper-parameter dictionary.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Regression deltas with shape
        `(batch_size, total_anchors, 4)` and classification labels reshaped to the
        feature map.
    """
    batch_size = tf.shape(gt_boxes)[0]
    feature_map_shape = hyper_params["feature_map_shape"]
    anchor_count = hyper_params["anchor_count"]
    total_pos_bboxes = hyper_params["total_pos_bboxes"]
    total_neg_bboxes = hyper_params["total_neg_bboxes"]
    variances = hyper_params["variances"]
    iou_map = bbox_utils.generate_iou_map(anchors, gt_boxes)
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    pos_mask = tf.greater(merged_iou_map, 0.7)
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    # Every valid ground-truth box keeps at least one positive anchor even when
    # its best IoU falls below the normal positive threshold.
    max_pos_mask = tf.scatter_nd(scatter_bbox_indices, tf.fill((tf.shape(valid_indices)[0],), True), tf.shape(pos_mask))
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    pos_mask = randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count
    # Negatives are sampled after positives so the final minibatch stays balanced.
    neg_mask = tf.logical_and(tf.less(merged_iou_map, 0.3), tf.logical_not(pos_mask))
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count)
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_row, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(anchors, expanded_gt_boxes) / variances
    bbox_labels = tf.reshape(bbox_labels, (batch_size, feature_map_shape, feature_map_shape, anchor_count))
    return bbox_deltas, bbox_labels


def frcnn_cls_loss(*args: Any) -> tf.Tensor:
    """Compute the masked classification loss for Faster R-CNN outputs.

    Args:
        *args (Any): Either `(y_true, y_pred)` or a single tuple containing both
            tensors.

    Returns:
        tf.Tensor: Scalar Faster R-CNN classification loss.
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    loss_fn = tf.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
    loss_for_all = loss_fn(y_true, y_pred)
    cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    mask = tf.cast(cond, dtype=tf.float32)
    conf_loss = tf.reduce_sum(mask * loss_for_all)
    total_boxes = tf.maximum(1.0, tf.reduce_sum(mask))
    return conf_loss / total_boxes


def rpn_cls_loss(*args: Any) -> tf.Tensor:
    """Compute the RPN objectness loss while ignoring neutral anchors.

    Args:
        *args (Any): Either `(y_true, y_pred)` or a single tuple containing both
            tensors.

    Returns:
        tf.Tensor: Scalar RPN classification loss.
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    indices = tf.where(tf.not_equal(y_true, tf.constant(-1.0, dtype=tf.float32)))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    loss_fn = tf.losses.BinaryCrossentropy()
    return loss_fn(target, output)


def reg_loss(*args: Any) -> tf.Tensor:
    """Compute the regression loss over positive proposals only.

    Args:
        *args (Any): Either `(y_true, y_pred)` or a single tuple containing both
            tensors.

    Returns:
        tf.Tensor: Scalar regression loss.
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, 4))
    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    loss_for_all = loss_fn(y_true, y_pred)
    loss_for_all = tf.reduce_sum(loss_for_all, axis=-1)
    pos_cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    loc_loss = tf.reduce_sum(pos_mask * loss_for_all)
    total_pos_bboxes = tf.maximum(1.0, tf.reduce_sum(pos_mask))
    return loc_loss / total_pos_bboxes
