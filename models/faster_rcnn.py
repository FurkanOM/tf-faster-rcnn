"""Faster R-CNN model components and assembly helpers."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Lambda, Layer, TimeDistributed
from tensorflow.keras.models import Model

from utils import bbox_utils, train_utils


HyperParams = Dict[str, Any]
DecodedPredictions = Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor"]
RoITargets = Tuple["tf.Tensor", "tf.Tensor"]


class Decoder(Layer):
    """Decode Faster R-CNN logits and deltas into final detections."""

    def __init__(
        self,
        variances: Sequence[float],
        total_labels: int,
        max_total_size: int = 200,
        score_threshold: float = 0.5,
        **kwargs: Any
    ) -> None:
        """Initialize the decoder layer.

        Args:
            variances (Sequence[float]): Bounding-box variance values.
            total_labels (int): Number of class labels including background.
            max_total_size (int): Maximum detections kept after NMS.
            score_threshold (float): Minimum score kept after decoding.
            **kwargs (Any): Additional Keras layer keyword arguments.

        Returns:
            None: The layer is initialized in place.
        """
        super(Decoder, self).__init__(**kwargs)
        self.variances = variances
        self.total_labels = total_labels
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable layer configuration.

        Returns:
            Dict[str, Any]: Keras serialization dictionary for this layer.
        """
        config = super(Decoder, self).get_config()
        config.update({
            "variances": self.variances,
            "total_labels": self.total_labels,
            "max_total_size": self.max_total_size,
            "score_threshold": self.score_threshold
        })
        return config

    def call(self, inputs: Sequence[tf.Tensor]) -> DecodedPredictions:
        """Decode Faster R-CNN class and box predictions into final detections.

        Args:
            inputs (Sequence[tf.Tensor]): RoI boxes, predicted deltas, and class
                probabilities.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Final boxes, predicted labels,
            and confidence scores.
        """
        roi_bboxes = inputs[0]
        pred_deltas = inputs[1]
        pred_label_probs = inputs[2]
        batch_size = tf.shape(pred_deltas)[0]
        pred_deltas = tf.reshape(pred_deltas, (batch_size, -1, self.total_labels, 4))
        pred_deltas *= self.variances
        expanded_roi_bboxes = tf.tile(tf.expand_dims(roi_bboxes, -2), (1, 1, self.total_labels, 1))
        pred_bboxes = bbox_utils.get_bboxes_from_deltas(expanded_roi_bboxes, pred_deltas)
        pred_labels_map = tf.expand_dims(tf.argmax(pred_label_probs, -1), -1)
        pred_labels = tf.where(tf.not_equal(pred_labels_map, 0), pred_label_probs, tf.zeros_like(pred_label_probs))
        final_bboxes, final_scores, final_labels, _ = bbox_utils.non_max_suppression(
            pred_bboxes,
            pred_labels,
            max_output_size_per_class=self.max_total_size,
            max_total_size=self.max_total_size,
            score_threshold=self.score_threshold
        )
        return final_bboxes, final_labels, final_scores


class RoIBBox(Layer):
    """Convert RPN outputs into a fixed number of region proposals."""

    def __init__(
        self,
        anchors: tf.Tensor,
        mode: str,
        hyper_params: HyperParams,
        **kwargs: Any
    ) -> None:
        """Initialize the RoI proposal layer.

        Args:
            anchors (tf.Tensor): Anchor tensor with shape `(total_anchors, 4)`.
            mode (str): Either `"training"` or `"inference"`.
            hyper_params (Dict[str, Any]): Hyper-parameter dictionary.
            **kwargs (Any): Additional Keras layer keyword arguments.

        Returns:
            None: The layer is initialized in place.
        """
        super(RoIBBox, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        self.mode = mode
        self.anchors = tf.constant(anchors, dtype=tf.float32)

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable layer configuration.

        Returns:
            Dict[str, Any]: Keras serialization dictionary for this layer.
        """
        config = super(RoIBBox, self).get_config()
        config.update({"hyper_params": self.hyper_params, "anchors": self.anchors.numpy(), "mode": self.mode})
        return config

    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        """Convert RPN outputs into a fixed number of region proposals.

        Args:
            inputs (Sequence[tf.Tensor]): RPN deltas and objectness logits.

        Returns:
            tf.Tensor: Proposed RoI boxes with shape `(batch_size, top_n, 4)`.
        """
        rpn_bbox_deltas = inputs[0]
        rpn_labels = inputs[1]
        pre_nms_topn = self.hyper_params["pre_nms_topn"]
        post_nms_topn = self.hyper_params["train_nms_topn"] if self.mode == "training" else self.hyper_params["test_nms_topn"]
        nms_iou_threshold = self.hyper_params["nms_iou_threshold"]
        variances = self.hyper_params["variances"]
        total_anchors = self.anchors.shape[0]
        batch_size = tf.shape(rpn_bbox_deltas)[0]
        rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, total_anchors, 4))
        rpn_labels = tf.reshape(rpn_labels, (batch_size, total_anchors))
        rpn_bbox_deltas *= variances
        rpn_bboxes = bbox_utils.get_bboxes_from_deltas(self.anchors, rpn_bbox_deltas)
        _, pre_indices = tf.nn.top_k(rpn_labels, pre_nms_topn)
        pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
        pre_roi_labels = tf.gather(rpn_labels, pre_indices, batch_dims=1)
        pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
        pre_roi_labels = tf.reshape(pre_roi_labels, (batch_size, pre_nms_topn, 1))
        roi_bboxes, _, _, _ = bbox_utils.non_max_suppression(
            pre_roi_bboxes,
            pre_roi_labels,
            max_output_size_per_class=post_nms_topn,
            max_total_size=post_nms_topn,
            iou_threshold=nms_iou_threshold
        )
        return tf.stop_gradient(roi_bboxes)


class RoIDelta(Layer):
    """Build Faster R-CNN regression and classification targets for sampled RoIs."""

    def __init__(self, hyper_params: HyperParams, **kwargs: Any) -> None:
        """Initialize the RoI target-generation layer.

        Args:
            hyper_params (Dict[str, Any]): Hyper-parameter dictionary.
            **kwargs (Any): Additional Keras layer keyword arguments.

        Returns:
            None: The layer is initialized in place.
        """
        super(RoIDelta, self).__init__(**kwargs)
        self.hyper_params = hyper_params

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable layer configuration.

        Returns:
            Dict[str, Any]: Keras serialization dictionary for this layer.
        """
        config = super(RoIDelta, self).get_config()
        config.update({"hyper_params": self.hyper_params})
        return config

    def call(self, inputs: Sequence[tf.Tensor]) -> RoITargets:
        """Build regression and classification targets for sampled RoIs.

        Args:
            inputs (Sequence[tf.Tensor]): Proposed RoIs, ground-truth boxes, and
                ground-truth labels.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Faster R-CNN regression and one-hot class
            targets for the sampled RoIs.
        """
        roi_bboxes = inputs[0]
        gt_boxes = inputs[1]
        gt_labels = inputs[2]
        total_labels = self.hyper_params["total_labels"]
        total_pos_bboxes = self.hyper_params["total_pos_bboxes"]
        total_neg_bboxes = self.hyper_params["total_neg_bboxes"]
        variances = self.hyper_params["variances"]
        batch_size = tf.shape(roi_bboxes)[0]
        total_bboxes = tf.shape(roi_bboxes)[1]
        iou_map = bbox_utils.generate_iou_map(roi_bboxes, gt_boxes)
        max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
        merged_iou_map = tf.reduce_max(iou_map, axis=2)
        pos_mask = tf.greater(merged_iou_map, 0.5)
        pos_mask = train_utils.randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
        neg_mask = tf.logical_and(tf.less(merged_iou_map, 0.5), tf.greater(merged_iou_map, 0.1))
        neg_mask = train_utils.randomly_select_xyz_mask(neg_mask, tf.constant([total_neg_bboxes], dtype=tf.int32))
        gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
        expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, axis=-1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
        gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
        pos_gt_labels = tf.where(pos_mask, gt_labels_map, tf.constant(-1, dtype=tf.int32))
        neg_gt_labels = tf.cast(neg_mask, dtype=tf.int32)
        expanded_gt_labels = pos_gt_labels + neg_gt_labels
        roi_bbox_deltas = bbox_utils.get_deltas_from_bboxes(roi_bboxes, expanded_gt_boxes) / variances
        roi_bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
        scatter_indices = tf.tile(tf.expand_dims(roi_bbox_labels, -1), (1, 1, 1, 4))
        roi_bbox_deltas = scatter_indices * tf.expand_dims(roi_bbox_deltas, -2)
        roi_bbox_deltas = tf.reshape(roi_bbox_deltas, (batch_size, total_bboxes * total_labels, 4))
        return tf.stop_gradient(roi_bbox_deltas), tf.stop_gradient(roi_bbox_labels)


class RoIPooling(Layer):
    """Crop and resize feature-map regions into a fixed pooling shape."""

    def __init__(self, hyper_params: HyperParams, **kwargs: Any) -> None:
        """Initialize the RoI pooling layer.

        Args:
            hyper_params (Dict[str, Any]): Hyper-parameter dictionary.
            **kwargs (Any): Additional Keras layer keyword arguments.

        Returns:
            None: The layer is initialized in place.
        """
        super(RoIPooling, self).__init__(**kwargs)
        self.hyper_params = hyper_params

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable layer configuration.

        Returns:
            Dict[str, Any]: Keras serialization dictionary for this layer.
        """
        config = super(RoIPooling, self).get_config()
        config.update({"hyper_params": self.hyper_params})
        return config

    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        """Crop and resize feature-map regions into a fixed pooling shape.

        Args:
            inputs (Sequence[tf.Tensor]): Feature map tensor and proposed RoIs.

        Returns:
            tf.Tensor: Pooled feature tensor with shape
            `(batch_size, total_bboxes, pooled_height, pooled_width, channels)`.
        """
        feature_map = inputs[0]
        roi_bboxes = inputs[1]
        pooling_size = self.hyper_params["pooling_size"]
        batch_size = tf.shape(roi_bboxes)[0]
        total_bboxes = tf.shape(roi_bboxes)[1]
        row_size = batch_size * total_bboxes
        pooling_bbox_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes))
        pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1,))
        pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))
        pooling_feature_map = tf.image.crop_and_resize(
            feature_map,
            pooling_bboxes,
            pooling_bbox_indices,
            pooling_size
        )
        final_pooling_feature_map = tf.reshape(
            pooling_feature_map,
            (batch_size, total_bboxes, pooling_feature_map.shape[1], pooling_feature_map.shape[2], pooling_feature_map.shape[3])
        )
        return final_pooling_feature_map


def get_model(
    feature_extractor: tf.keras.layers.Layer,
    rpn_model: tf.keras.Model,
    anchors: tf.Tensor,
    hyper_params: HyperParams,
    mode: str = "training"
) -> tf.keras.Model:
    """Assemble the Faster R-CNN model for training or inference.

    Args:
        feature_extractor (tf.keras.layers.Layer): Backbone feature extractor
            layer.
        rpn_model (tf.keras.Model): Prebuilt RPN model.
        anchors (tf.Tensor): Anchor tensor with shape `(total_anchors, 4)`.
        hyper_params (Dict[str, Any]): Hyper-parameter dictionary.
        mode (str): Either `"training"` or `"inference"`.

    Returns:
        tf.keras.Model: Configured Faster R-CNN model.
    """
    input_img = rpn_model.input
    rpn_reg_predictions, rpn_cls_predictions = rpn_model.output
    roi_bboxes = RoIBBox(anchors, mode, hyper_params, name="roi_bboxes")([rpn_reg_predictions, rpn_cls_predictions])
    roi_pooled = RoIPooling(hyper_params, name="roi_pooling")([feature_extractor.output, roi_bboxes])
    output = TimeDistributed(Flatten(), name="frcnn_flatten")(roi_pooled)
    output = TimeDistributed(Dense(4096, activation="relu"), name="frcnn_fc1")(output)
    output = TimeDistributed(Dropout(0.5), name="frcnn_dropout1")(output)
    output = TimeDistributed(Dense(4096, activation="relu"), name="frcnn_fc2")(output)
    output = TimeDistributed(Dropout(0.5), name="frcnn_dropout2")(output)
    frcnn_cls_predictions = TimeDistributed(Dense(hyper_params["total_labels"], activation="softmax"), name="frcnn_cls")(output)
    frcnn_reg_predictions = TimeDistributed(Dense(hyper_params["total_labels"] * 4, activation="linear"), name="frcnn_reg")(output)
    if mode == "training":
        input_gt_boxes = Input(shape=(None, 4), name="input_gt_boxes", dtype=tf.float32)
        input_gt_labels = Input(shape=(None,), name="input_gt_labels", dtype=tf.int32)
        rpn_cls_actuals = Input(shape=(None, None, hyper_params["anchor_count"]), name="input_rpn_cls_actuals", dtype=tf.float32)
        rpn_reg_actuals = Input(shape=(None, 4), name="input_rpn_reg_actuals", dtype=tf.float32)
        frcnn_reg_actuals, frcnn_cls_actuals = RoIDelta(hyper_params, name="roi_deltas")(
            [roi_bboxes, input_gt_boxes, input_gt_labels]
        )
        loss_names = ["rpn_reg_loss", "rpn_cls_loss", "frcnn_reg_loss", "frcnn_cls_loss"]
        rpn_reg_loss_layer = Lambda(train_utils.reg_loss, name=loss_names[0])([rpn_reg_actuals, rpn_reg_predictions])
        rpn_cls_loss_layer = Lambda(train_utils.rpn_cls_loss, name=loss_names[1])([rpn_cls_actuals, rpn_cls_predictions])
        frcnn_reg_loss_layer = Lambda(train_utils.reg_loss, name=loss_names[2])([frcnn_reg_actuals, frcnn_reg_predictions])
        frcnn_cls_loss_layer = Lambda(train_utils.frcnn_cls_loss, name=loss_names[3])([frcnn_cls_actuals, frcnn_cls_predictions])
        frcnn_model = Model(
            inputs=[input_img, input_gt_boxes, input_gt_labels, rpn_reg_actuals, rpn_cls_actuals],
            outputs=[
                roi_bboxes,
                rpn_reg_predictions,
                rpn_cls_predictions,
                frcnn_reg_predictions,
                frcnn_cls_predictions,
                rpn_reg_loss_layer,
                rpn_cls_loss_layer,
                frcnn_reg_loss_layer,
                frcnn_cls_loss_layer,
            ]
        )
        for layer_name in loss_names:
            layer = frcnn_model.get_layer(layer_name)
            frcnn_model.add_loss(layer.output)
            frcnn_model.add_metric(layer.output, name=layer_name, aggregation="mean")
    else:
        bboxes, labels, scores = Decoder(
            hyper_params["variances"],
            hyper_params["total_labels"],
            name="faster_rcnn_decoder"
        )([roi_bboxes, frcnn_reg_predictions, frcnn_cls_predictions])
        frcnn_model = Model(inputs=input_img, outputs=[bboxes, labels, scores])
    return frcnn_model


def init_model(model: tf.keras.Model, hyper_params: HyperParams) -> None:
    """Warm up the training graph with dummy inputs and targets.

    Args:
        model (tf.keras.Model): Faster R-CNN model instance to initialize.
        hyper_params (Dict[str, Any]): Hyper-parameter dictionary.

    Returns:
        None: The model graph is built in place.
    """
    final_height = hyper_params["img_size"]
    final_width = hyper_params["img_size"]
    img = tf.random.uniform((1, final_height, final_width, 3))
    feature_map_shape = hyper_params["feature_map_shape"]
    total_anchors = feature_map_shape * feature_map_shape * hyper_params["anchor_count"]
    gt_boxes = tf.random.uniform((1, 1, 4))
    gt_labels = tf.random.uniform((1, 1), maxval=hyper_params["total_labels"], dtype=tf.int32)
    bbox_deltas = tf.random.uniform((1, total_anchors, 4))
    bbox_labels = tf.random.uniform((1, feature_map_shape, feature_map_shape, hyper_params["anchor_count"]), maxval=1, dtype=tf.float32)
    model([img, gt_boxes, gt_labels, bbox_deltas, bbox_labels])
