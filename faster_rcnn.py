import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Lambda, Input, Conv2D, TimeDistributed, Dense, Flatten, BatchNormalization, Dropout
import numpy as np
import helpers
import rpn

def roibbox(anchors, hyper_params, rpn_bbox_deltas, rpn_labels):
    pre_nms_topn = hyper_params["pre_nms_topn"]
    post_nms_topn = hyper_params["post_nms_topn"]
    nms_iou_threshold = hyper_params["nms_iou_threshold"]
    total_anchors = anchors.shape[0]
    batch_size = tf.shape(rpn_bbox_deltas)[0]
    rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, total_anchors, 4))
    rpn_labels = tf.reshape(rpn_labels, (batch_size, total_anchors))
    rpn_bboxes = helpers.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)
    #
    _, pre_indices = tf.nn.top_k(rpn_labels, pre_nms_topn)
    #
    pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
    pre_roi_labels = tf.gather(rpn_labels, pre_indices, batch_dims=1)
    #
    pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
    pre_roi_labels = tf.reshape(pre_roi_labels, (batch_size, pre_nms_topn, 1))
    #
    roi_bboxes, _, _, _ = helpers.non_max_suppression(pre_roi_bboxes, pre_roi_labels,
                                                      max_output_size_per_class=post_nms_topn,
                                                      max_total_size=post_nms_topn,
                                                      iou_threshold=nms_iou_threshold)
    return roi_bboxes

def roidelta(roi_bboxes, gt_boxes, gt_labels, hyper_params):
    total_labels = hyper_params["total_labels"]
    total_pos_bboxes = hyper_params["total_pos_bboxes"]
    batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = helpers.generate_iou_map(roi_bboxes, gt_boxes)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    # Sorted iou values
    sorted_iou_map = tf.argsort(merged_iou_map, direction="DESCENDING")
    # Sort indices for generating masks
    sorted_map_indices = tf.argsort(sorted_iou_map)
    # Generate pos mask for pos bboxes
    pos_mask = tf.less(sorted_map_indices, total_pos_bboxes)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, axis=-1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    #
    gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_labels = tf.where(pos_mask, gt_labels_map, tf.zeros_like(gt_labels_map))
    #
    roi_bbox_deltas = helpers.get_deltas_from_bboxes(roi_bboxes, expanded_gt_boxes)
    #
    roi_bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
    scatter_indices = tf.tile(tf.expand_dims(roi_bbox_labels, -1), (1, 1, 1, 4))
    roi_bbox_deltas = scatter_indices * tf.expand_dims(roi_bbox_deltas, -2)
    roi_bbox_deltas = tf.reshape(roi_bbox_deltas, (batch_size, total_bboxes, total_labels * 4))
    return roi_bbox_deltas, roi_bbox_labels

def roipooling(feature_map, roi_bboxes, hyper_params):
    pooling_size = hyper_params["pooling_size"]
    batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
    #
    row_size = batch_size * total_bboxes
    # We need to arange bbox indices for each batch
    pooling_bbox_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes))
    pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1, ))
    pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))
    # Crop to bounding box size then resize to pooling size
    pooling_feature_map = tf.image.crop_and_resize(
        feature_map,
        pooling_bboxes,
        pooling_bbox_indices,
        pooling_size
    )
    final_pooling_feature_map = tf.reshape(pooling_feature_map, (batch_size, total_bboxes, pooling_feature_map.shape[1], pooling_feature_map.shape[2], pooling_feature_map.shape[3]))
    return final_pooling_feature_map

class RoIBBox(Layer):
    """Generating bounding boxes from rpn predictions.
    First calculating the boxes from predicted deltas and label probs.
    Then applied non max suppression and selecting "post_nms_topn" boxes.
    inputs:
        rpn_bbox_deltas = (batch_size, img_output_height, img_output_width, anchor_count * [delta_y, delta_x, delta_h, delta_w])
            img_output_height and img_output_width are calculated to the base model output
            they are (img_height // stride) and (img_width // stride) for VGG16 backbone
        rpn_labels = (batch_size, img_output_height, img_output_width, anchor_count)

    outputs:
        roi_bboxes = (batch_size, post_nms_topn, [y1, x1, y2, x2])
    """

    def __init__(self, anchors, hyper_params, **kwargs):
        super(RoIBBox, self).__init__(**kwargs)
        self.hyper_params = hyper_params
        self.anchors = tf.constant(anchors, dtype=tf.float32)

    def get_config(self):
        config = super(RoIBBox, self).get_config()
        config.update({"hyper_params": self.hyper_params, "anchors": self.anchors})
        return config

    def call(self, inputs):
        rpn_bbox_deltas = inputs[0]
        rpn_labels = inputs[1]
        anchors = self.anchors
        #
        pre_nms_topn = self.hyper_params["pre_nms_topn"]
        post_nms_topn = self.hyper_params["post_nms_topn"]
        nms_iou_threshold = self.hyper_params["nms_iou_threshold"]
        total_anchors = anchors.shape[0]
        batch_size = tf.shape(rpn_bbox_deltas)[0]
        rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, total_anchors, 4))
        rpn_labels = tf.reshape(rpn_labels, (batch_size, total_anchors))
        rpn_bboxes = helpers.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)
        #
        _, pre_indices = tf.nn.top_k(rpn_labels, pre_nms_topn)
        #
        pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
        pre_roi_labels = tf.gather(rpn_labels, pre_indices, batch_dims=1)
        #
        pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
        pre_roi_labels = tf.reshape(pre_roi_labels, (batch_size, pre_nms_topn, 1))
        #
        roi_bboxes, _, _, _ = helpers.non_max_suppression(pre_roi_bboxes, pre_roi_labels,
                                                          max_output_size_per_class=post_nms_topn,
                                                          max_total_size=post_nms_topn,
                                                          iou_threshold=nms_iou_threshold)
        #
        return tf.stop_gradient(roi_bboxes)

class RoIDelta(Layer):
    """Calculating faster rcnn actual bounding box deltas and labels.
    This layer only running on the training phase.
    inputs:
        roi_bboxes = (batch_size, nms_topn, [y1, x1, y2, x2])
        gt_boxes = (batch_size, padded_gt_boxes_size, [y1, x1, y2, x2])
        gt_labels = (batch_size, padded_gt_boxes_size)

    outputs:
        roi_bbox_deltas = (batch_size, nms_topn, total_labels * [delta_y, delta_x, delta_h, delta_w])
        roi_bbox_labels = (batch_size, nms_topn, total_labels)
    """

    def __init__(self, hyper_params, **kwargs):
        super(RoIDelta, self).__init__(**kwargs)
        self.hyper_params = hyper_params

    def get_config(self):
        config = super(RoIDelta, self).get_config()
        config.update({"hyper_params": self.hyper_params})
        return config

    def call(self, inputs):
        roi_bboxes = inputs[0]
        gt_boxes = inputs[1]
        gt_labels = inputs[2]
        total_labels = self.hyper_params["total_labels"]
        total_pos_bboxes = self.hyper_params["total_pos_bboxes"]
        batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
        # Calculate iou values between each bboxes and ground truth boxes
        iou_map = helpers.generate_iou_map(roi_bboxes, gt_boxes)
        # Get max index value for each row
        max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
        # IoU map has iou values for every gt boxes and we merge these values column wise
        merged_iou_map = tf.reduce_max(iou_map, axis=2)
        # Sorted iou values
        sorted_iou_map = tf.argsort(merged_iou_map, direction="DESCENDING")
        # Sort indices for generating masks
        sorted_map_indices = tf.argsort(sorted_iou_map)
        # Generate pos mask for pos bboxes
        pos_mask = tf.less(sorted_map_indices, total_pos_bboxes)
        #
        gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
        expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, axis=-1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
        #
        gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
        expanded_gt_labels = tf.where(pos_mask, gt_labels_map, tf.zeros_like(gt_labels_map))
        #
        roi_bbox_deltas = helpers.get_deltas_from_bboxes(roi_bboxes, expanded_gt_boxes)
        #
        roi_bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
        scatter_indices = tf.tile(tf.expand_dims(roi_bbox_labels, -1), (1, 1, 1, 4))
        roi_bbox_deltas = scatter_indices * tf.expand_dims(roi_bbox_deltas, -2)
        roi_bbox_deltas = tf.reshape(roi_bbox_deltas, (batch_size, total_bboxes * total_labels, 4))
        #
        return tf.stop_gradient(roi_bbox_deltas), tf.stop_gradient(roi_bbox_labels)

class RoIPooling(Layer):
    """Reducing all feature maps to same size.
    Firstly cropping bounding boxes from the feature maps and then resizing it to the pooling size.
    inputs:
        feature_map = (batch_size, img_output_height, img_output_width, channels)
        roi_bboxes = (batch_size, nms_topn, [y1, x1, y2, x2])

    outputs:
        final_pooling_feature_map = (batch_size, nms_topn, pooling_size[0], pooling_size[1], channels)
            pooling_size usually (7, 7)
    """

    def __init__(self, hyper_params, **kwargs):
        super(RoIPooling, self).__init__(**kwargs)
        self.hyper_params = hyper_params

    def get_config(self):
        config = super(RoIPooling, self).get_config()
        config.update({"hyper_params": self.hyper_params})
        return config

    def call(self, inputs):
        feature_map = inputs[0]
        roi_bboxes = inputs[1]
        pooling_size = self.hyper_params["pooling_size"]
        batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
        #
        row_size = batch_size * total_bboxes
        # We need to arange bbox indices for each batch
        pooling_bbox_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes))
        pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1, ))
        pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))
        # Crop to bounding box size then resize to pooling size
        pooling_feature_map = tf.image.crop_and_resize(
            feature_map,
            pooling_bboxes,
            pooling_bbox_indices,
            pooling_size
        )
        final_pooling_feature_map = tf.reshape(pooling_feature_map, (batch_size, total_bboxes, pooling_feature_map.shape[1], pooling_feature_map.shape[2], pooling_feature_map.shape[3]))
        return final_pooling_feature_map

def generator(dataset, anchors, hyper_params, input_processor):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary
        input_processor = function for preparing image for input. It's getting from backbone.

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            _, gt_boxes, gt_labels = image_data
            input_img, bbox_deltas, bbox_labels = rpn.get_step_data(image_data, anchors, hyper_params, input_processor)
            yield (input_img, gt_boxes, gt_labels, bbox_deltas, bbox_labels), ()

def get_model(base_model, rpn_model, anchors, hyper_params, mode="training"):
    """Generating rpn model for given backbone base model and hyper params.
    inputs:
        base_model = tf.keras.model pretrained backbone, only VGG16 available for now
        rpn_model = tf.keras.model generated rpn model
        hyper_params = dictionary
        mode = "training" or "inference"

    outputs:
        frcnn_model = tf.keras.model
    """
    input_img = base_model.input
    rpn_reg_predictions, rpn_cls_predictions = rpn_model.output
    #
    roi_bboxes = RoIBBox(anchors, hyper_params, name="roi_bboxes")([rpn_reg_predictions, rpn_cls_predictions])
    #
    roi_pooled = RoIPooling(hyper_params, name="roi_pooling")([base_model.output, roi_bboxes])
    #
    output = TimeDistributed(Flatten(), name="frcnn_flatten")(roi_pooled)
    output = TimeDistributed(Dense(4096, activation="relu"), name="frcnn_fc1")(output)
    output = TimeDistributed(BatchNormalization(), name="frcnn_batch_norm1")(output)
    output = TimeDistributed(Dropout(0.2), name="frcnn_dropout1")(output)
    output = TimeDistributed(Dense(2048, activation="relu"), name="frcnn_fc2")(output)
    output = TimeDistributed(BatchNormalization(), name="frcnn_batch_norm2")(output)
    output = TimeDistributed(Dropout(0.2), name="frcnn_dropout2")(output)
    frcnn_cls_predictions = TimeDistributed(Dense(hyper_params["total_labels"], activation="softmax"), name="frcnn_cls")(output)
    frcnn_reg_predictions = TimeDistributed(Dense(hyper_params["total_labels"] * 4, activation="linear"), name="frcnn_reg")(output)
    #
    if mode == "training":
        input_gt_boxes = Input(shape=(None, 4), name="input_gt_boxes", dtype=tf.float32)
        input_gt_labels = Input(shape=(None, ), name="input_gt_labels", dtype=tf.int32)
        rpn_cls_actuals = Input(shape=(None, None, hyper_params["anchor_count"]), name="input_rpn_cls_actuals", dtype=tf.float32)
        rpn_reg_actuals = Input(shape=(None, 4), name="input_rpn_reg_actuals", dtype=tf.float32)
        frcnn_reg_actuals, frcnn_cls_actuals = RoIDelta(hyper_params, name="roi_deltas")(
                                                        [roi_bboxes, input_gt_boxes, input_gt_labels])
        #
        loss_names = ["rpn_reg_loss", "rpn_cls_loss", "frcnn_reg_loss", "frcnn_cls_loss"]
        rpn_reg_loss_layer = Lambda(helpers.reg_loss, name=loss_names[0])([rpn_reg_actuals, rpn_reg_predictions])
        rpn_cls_loss_layer = Lambda(helpers.rpn_cls_loss, name=loss_names[1])([rpn_cls_actuals, rpn_cls_predictions])
        frcnn_reg_loss_layer = Lambda(helpers.reg_loss, name=loss_names[2])([frcnn_reg_actuals, frcnn_reg_predictions])
        frcnn_cls_loss_layer = Lambda(helpers.frcnn_cls_loss, name=loss_names[3])([frcnn_cls_actuals, frcnn_cls_predictions])
        #
        frcnn_model = Model(inputs=[input_img, input_gt_boxes, input_gt_labels,
                              rpn_reg_actuals, rpn_cls_actuals],
                      outputs=[roi_bboxes, rpn_reg_predictions, rpn_cls_predictions,
                               frcnn_reg_predictions, frcnn_cls_predictions,
                               rpn_reg_loss_layer, rpn_cls_loss_layer,
                               frcnn_reg_loss_layer, frcnn_cls_loss_layer])
        #
        for layer_name in loss_names:
            layer = frcnn_model.get_layer(layer_name)
            frcnn_model.add_loss(layer.output)
            frcnn_model.add_metric(layer.output, name=layer_name, aggregation="mean")
        #
    else:
        frcnn_model = Model(inputs=[input_img],
                      outputs=[roi_bboxes, rpn_reg_predictions, rpn_cls_predictions,
                               frcnn_reg_predictions, frcnn_cls_predictions])
        #
    return frcnn_model

def init_model(model, hyper_params):
    """Generating dummy data for initialize model.
    In this way, the training process can continue from where it left off.
    inputs:
        model = tf.keras.model
        hyper_params = dictionary
    """
    final_height, final_width = helpers.VOC["max_height"], helpers.VOC["max_width"]
    img = tf.random.uniform((1, final_height, final_width, 3))
    output_height, output_width = final_height // hyper_params["stride"], final_width // hyper_params["stride"]
    total_anchors = output_height * output_width * hyper_params["anchor_count"]
    gt_boxes = tf.random.uniform((1, 1, 4))
    gt_labels = tf.random.uniform((1, 1), maxval=hyper_params["total_labels"], dtype=tf.int32)
    bbox_deltas = tf.random.uniform((1, output_height*output_width*hyper_params["anchor_count"], 4))
    bbox_labels = tf.random.uniform((1, output_height, output_width, hyper_params["anchor_count"]), maxval=1, dtype=tf.float32)
    model([img, gt_boxes, gt_labels, bbox_deltas, bbox_labels])
