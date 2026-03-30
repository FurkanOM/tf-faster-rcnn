"""Evaluation helpers for mean average precision reporting."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import tensorflow as tf

from utils import bbox_utils


ClassStats = Dict[str, Any]
StatsDict = Dict[int, ClassStats]


def init_stats(labels: Sequence[str]) -> StatsDict:
    """Initialize per-class statistics containers, skipping background.

    Args:
        labels (Sequence[str]): Ordered label names, including background at index
            `0`.

    Returns:
        Dict[int, Dict[str, Any]]: Per-class counters and score buffers keyed by
        label index.
    """
    stats: StatsDict = {}
    for i, label in enumerate(labels):
        if i == 0:
            continue
        stats[i] = {
            "label": label,
            "total": 0,
            "tp": [],
            "fp": [],
            "scores": [],
        }
    return stats


def update_stats(
    pred_bboxes: tf.Tensor,
    pred_labels: tf.Tensor,
    pred_scores: tf.Tensor,
    gt_boxes: tf.Tensor,
    gt_labels: tf.Tensor,
    stats: StatsDict
) -> StatsDict:
    """Accumulate true and false positive counts for a prediction batch.

    Args:
        pred_bboxes (tf.Tensor): Predicted bounding boxes with shape
            `(batch_size, total_bboxes, 4)`.
        pred_labels (tf.Tensor): Predicted label indices with shape
            `(batch_size, total_bboxes)`.
        pred_scores (tf.Tensor): Predicted confidence scores with shape
            `(batch_size, total_bboxes)`.
        gt_boxes (tf.Tensor): Ground-truth boxes with shape
            `(batch_size, total_gt_boxes, 4)`.
        gt_labels (tf.Tensor): Ground-truth labels with shape
            `(batch_size, total_gt_boxes)`.
        stats (Dict[int, Dict[str, Any]]): Mutable per-class statistics
            container.

    Returns:
        Dict[int, Dict[str, Any]]: Updated statistics container.
    """
    iou_map = bbox_utils.generate_iou_map(pred_bboxes, gt_boxes)
    merged_iou_map = tf.reduce_max(iou_map, axis=-1)
    max_indices_each_gt = tf.argmax(iou_map, axis=-1, output_type=tf.int32)
    sorted_ids = tf.argsort(merged_iou_map, direction="DESCENDING")
    count_holder = tf.unique_with_counts(tf.reshape(gt_labels, (-1,)))
    for i, gt_label in enumerate(count_holder[0]):
        if gt_label == -1:
            continue
        gt_label = int(gt_label)
        stats[gt_label]["total"] += int(count_holder[2][i])
    for batch_id, _ in enumerate(merged_iou_map):
        true_labels = []
        for _, sorted_id in enumerate(sorted_ids[batch_id]):
            pred_label = pred_labels[batch_id, sorted_id]
            if pred_label == 0:
                continue
            iou = merged_iou_map[batch_id, sorted_id]
            gt_id = max_indices_each_gt[batch_id, sorted_id]
            gt_label = int(gt_labels[batch_id, gt_id])
            pred_label = int(pred_label)
            score = pred_scores[batch_id, sorted_id]
            stats[pred_label]["scores"].append(score)
            stats[pred_label]["tp"].append(0)
            stats[pred_label]["fp"].append(0)
            if iou >= 0.5 and pred_label == gt_label and gt_id not in true_labels:
                stats[pred_label]["tp"][-1] = 1
                true_labels.append(gt_id)
            else:
                stats[pred_label]["fp"][-1] = 1
    return stats


def calculate_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Calculate the 11-point interpolated average precision.

    Args:
        recall (np.ndarray): Recall values ordered by descending score.
        precision (np.ndarray): Precision values ordered by descending score.

    Returns:
        float: Average precision computed with 11-point interpolation.
    """
    ap = 0.0
    for recall_threshold in np.arange(0, 1.1, 0.1):
        prec_rec = precision[recall >= recall_threshold]
        if len(prec_rec) > 0:
            ap += np.amax(prec_rec)
    ap /= 11
    return ap


def calculate_mAP(stats: StatsDict) -> Tuple[StatsDict, float]:
    """Compute per-class AP values and the mean AP across classes.

    Args:
        stats (Dict[int, Dict[str, Any]]): Per-class statistics accumulated over
            the dataset.

    Returns:
        Tuple[Dict[int, Dict[str, Any]], float]: Updated statistics containing AP,
        recall, and precision arrays, plus the mean AP score.
    """
    aps = []
    for label in stats:
        label_stats = stats[label]
        tp = np.array(label_stats["tp"])
        fp = np.array(label_stats["fp"])
        scores = np.array(label_stats["scores"])
        ids = np.argsort(-scores)
        total = label_stats["total"]
        accumulated_tp = np.cumsum(tp[ids])
        accumulated_fp = np.cumsum(fp[ids])
        recall = accumulated_tp / total
        precision = accumulated_tp / (accumulated_fp + accumulated_tp)
        ap = calculate_ap(recall, precision)
        stats[label]["recall"] = recall
        stats[label]["precision"] = precision
        stats[label]["AP"] = ap
        aps.append(ap)
    mean_ap = np.mean(aps)
    return stats, mean_ap


def evaluate_predictions(
    dataset: tf.data.Dataset,
    pred_bboxes: tf.Tensor,
    pred_labels: tf.Tensor,
    pred_scores: tf.Tensor,
    labels: Sequence[str],
    batch_size: int
) -> StatsDict:
    """Evaluate model predictions over a dataset and print the resulting mAP.

    Args:
        dataset (tf.data.Dataset): Dataset yielding images and ground-truth
            annotations.
        pred_bboxes (tf.Tensor): Predicted bounding boxes for the full dataset.
        pred_labels (tf.Tensor): Predicted label indices for the full dataset.
        pred_scores (tf.Tensor): Predicted confidence scores for the full dataset.
        labels (Sequence[str]): Ordered label names including background.
        batch_size (int): Batch size used during prediction.

    Returns:
        Dict[int, Dict[str, Any]]: Final per-class evaluation statistics.
    """
    stats = init_stats(labels)
    for batch_id, image_data in enumerate(dataset):
        _, gt_boxes, gt_labels = image_data
        start = batch_id * batch_size
        end = start + batch_size
        batch_bboxes = pred_bboxes[start:end]
        batch_labels = pred_labels[start:end]
        batch_scores = pred_scores[start:end]
        stats = update_stats(batch_bboxes, batch_labels, batch_scores, gt_boxes, gt_labels, stats)
    stats, mean_ap = calculate_mAP(stats)
    print("mAP: {}".format(float(mean_ap)))
    return stats
