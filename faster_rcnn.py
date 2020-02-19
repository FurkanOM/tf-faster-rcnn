import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Lambda, Input, Conv2D, TimeDistributed, Dense, Flatten, BatchNormalization, Dropout
import numpy as np
import helpers
import rpn

def frcnn_cls_loss(args):
    y_true, y_pred = args
    lf = tf.losses.CategoricalCrossentropy()
    return tf.reduce_mean(lf(y_true, y_pred))

def rpn_cls_loss(args):
    y_true, y_pred = args
    indices = tf.where(tf.not_equal(y_true, -1))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.BinaryCrossentropy()
    return tf.reduce_mean(lf(target, output))

def reg_loss(args):
    y_true, y_pred = args
    indices = tf.where(tf.not_equal(y_true, 0))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    # Same with the smooth l1 loss
    lf = tf.losses.Huber()
    return tf.reduce_mean(lf(target, output))

def non_max_suppression(pred_bboxes, pred_labels, hyper_params):
    nms_bboxes, nms_scores, nms_labels, valid_detections = tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        hyper_params["nms_topn"],
        hyper_params["nms_topn"]
    )
    return nms_bboxes

def get_bboxes_from_deltas(anchors, deltas):
    all_anc_width = anchors[:, :, 3] - anchors[:, :, 1]
    all_anc_height = anchors[:, :, 2] - anchors[:, :, 0]
    all_anc_ctr_x = anchors[:, :, 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[:, :, 0] + 0.5 * all_anc_height
    #
    all_bbox_width = tf.exp(deltas[:, :, 3]) * all_anc_width
    all_bbox_height = tf.exp(deltas[:, :, 2]) * all_anc_height
    all_bbox_ctr_x = (deltas[:, :, 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (deltas[:, :, 0] * all_anc_height) + all_anc_ctr_y
    #
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    #
    return tf.stack([y1, x1, y2, x2], axis=2)

class RPNtoRoI(Layer):
    def __init__(self, hyper_params):
        super(RPNtoRoI, self).__init__()
        self.hyper_params = hyper_params

    def call(self, inputs):
        feature_map = inputs[0]
        rpn_bbox_deltas = inputs[1]
        rpn_labels = inputs[2]
        anchors = inputs[3]
        gt_boxes = inputs[4]
        gt_labels = inputs[5]
        ##############
        total_labels = self.hyper_params["total_labels"]
        total_pos_bboxes = self.hyper_params["total_pos_bboxes"]
        total_neg_bboxes = self.hyper_params["total_neg_bboxes"]
        pooling_size = self.hyper_params["pooling_size"]
        total_bboxes = total_pos_bboxes + total_neg_bboxes
        anchors_shape = tf.shape(anchors)
        batch_size, anchor_row_size = anchors_shape[0], anchors_shape[1]
        rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, anchor_row_size, 4))
        rpn_labels = tf.reshape(rpn_labels, (batch_size, anchor_row_size, 1))
        #
        rpn_bboxes = get_bboxes_from_deltas(anchors, rpn_bbox_deltas)
        rpn_bboxes = tf.reshape(rpn_bboxes, (batch_size, anchor_row_size, 1, 4))
        nms_bboxes = non_max_suppression(rpn_bboxes, rpn_labels, self.hyper_params)
        # Like in the RPN we calculate iou values
        # then pos-neg anchors / bboxes for non max suppressed bboxes using these values
        batch_total_pos_bboxes = tf.tile([total_pos_bboxes], (batch_size,))
        batch_total_neg_bboxes = tf.tile([total_neg_bboxes], (batch_size,))
        bbox_indices, gt_box_indices = tf.map_fn(helpers.get_selected_indices,
                                                (nms_bboxes, gt_boxes, batch_total_pos_bboxes, batch_total_neg_bboxes),
                                                dtype=(tf.int32, tf.int32), swap_memory=True)
        ################################################################
        roi_bboxes = tf.gather(nms_bboxes, bbox_indices, batch_dims=1)
        roi_bboxes = tf.stop_gradient(roi_bboxes)
        # Pooling
        row_size = batch_size * total_bboxes
        # We need to arange bbox indices for each batch
        flatted_batch_indices = helpers.get_tiled_indices(batch_size, total_bboxes)
        pooling_bbox_indices = tf.reshape(flatted_batch_indices, (-1, ))
        pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))
        # Crop to bounding box size then resize to pooling size
        pooling_feature_map = tf.image.crop_and_resize(
            feature_map,
            pooling_bboxes,
            pooling_bbox_indices,
            pooling_size
        )
        final_pooling_feature_map = tf.reshape(pooling_feature_map, (batch_size, total_bboxes, pooling_feature_map.shape[1], pooling_feature_map.shape[2], pooling_feature_map.shape[3]))
        ####################################################################################################
        # If process is not training we don't need to apply below operations
        ####################################################################################################
        gt_boxes_map = helpers.get_gt_boxes_map(gt_boxes, gt_box_indices, batch_size, total_neg_bboxes)
        #
        pos_gt_labels_map = tf.gather(gt_labels, gt_box_indices, batch_dims=1)
        neg_gt_labels_map = tf.fill((batch_size, total_neg_bboxes), total_labels-1)
        gt_labels_map = tf.concat([pos_gt_labels_map, neg_gt_labels_map], axis=1)
        #
        roi_bbox_deltas = helpers.get_deltas_from_bboxes(roi_bboxes, gt_boxes_map)
        #
        flatted_bbox_indices = tf.reshape(tf.tile(tf.range(total_bboxes), (batch_size, )), (-1, 1))
        flatted_gt_labels_indices = tf.reshape(gt_labels_map, (-1, 1))
        scatter_indices = helpers.get_scatter_indices_for_bboxes([flatted_batch_indices, flatted_bbox_indices, flatted_gt_labels_indices], batch_size, total_bboxes)
        roi_bbox_deltas = tf.scatter_nd(scatter_indices, roi_bbox_deltas, (batch_size, total_bboxes, total_labels, 4))
        roi_bbox_deltas = tf.reshape(roi_bbox_deltas, (batch_size, total_bboxes, total_labels * 4))
        roi_bbox_labels = tf.scatter_nd(scatter_indices, tf.ones((batch_size, total_bboxes), tf.int32), (batch_size, total_bboxes, total_labels))
        #
        return final_pooling_feature_map, tf.stop_gradient(roi_bbox_deltas), tf.stop_gradient(roi_bbox_labels)

def generator(dataset, hyper_params, input_processor):
    while True:
        for image_data in dataset:
            _, gt_boxes, gt_labels = image_data
            input_img, bbox_deltas, bbox_labels, anchors = rpn.get_step_data(image_data, hyper_params, input_processor)
            yield (input_img, bbox_deltas, bbox_labels, anchors, gt_boxes, gt_labels), ()

def get_model(base_model, rpn_model, hyper_params):
    input_img = base_model.input
    rpn_reg_predictions, rpn_cls_predictions = rpn_model.output
    #
    input_anchors = Input(shape=(None, 4), name="input_anchors", dtype=tf.float32)
    input_gt_boxes = Input(shape=(None, 4), name="input_gt_boxes", dtype=tf.float32)
    input_gt_labels = Input(shape=(None, ), name="input_gt_labels", dtype=tf.int32)
    rpn_cls_actuals = Input(shape=(None, None, hyper_params["anchor_count"]), name="input_rpn_cls_actuals", dtype=tf.int32)
    rpn_reg_actuals = Input(shape=(None, None, hyper_params["anchor_count"] * 4), name="input_rpn_reg_actuals", dtype=tf.float32)
    #
    roi_output, frcnn_reg_actuals, frcnn_cls_actuals = RPNtoRoI(hyper_params)([
        base_model.output, rpn_reg_predictions, rpn_cls_predictions,
        input_anchors, input_gt_boxes, input_gt_labels])
    #
    output = TimeDistributed(Flatten(), name="frcnn_flatten")(roi_output)
    output = TimeDistributed(Dense(4096, activation="relu"), name="frcnn_fc1")(output)
    output = TimeDistributed(BatchNormalization(), name="frcnn_batch_norm")(output)
    output = TimeDistributed(Dropout(0.2), name="frcnn_dropout")(output)
    frcnn_cls_predictions = TimeDistributed(Dense(hyper_params["total_labels"], activation="softmax"), name="frcnn_cls")(output)
    frcnn_reg_predictions = TimeDistributed(Dense(hyper_params["total_labels"] * 4, activation="linear"), name="frcnn_reg")(output)
    #
    loss_names = ["rpn_reg_loss", "rpn_cls_loss", "frcnn_reg_loss", "frcnn_cls_loss"]
    rpn_reg_loss_layer = Lambda(reg_loss, name=loss_names[0])([rpn_reg_actuals, rpn_reg_predictions])
    rpn_cls_loss_layer = Lambda(rpn_cls_loss, name=loss_names[1])([rpn_cls_actuals, rpn_cls_predictions])
    frcnn_reg_loss_layer = Lambda(reg_loss, name=loss_names[2])([frcnn_reg_actuals, frcnn_reg_predictions])
    frcnn_cls_loss_layer = Lambda(frcnn_cls_loss, name=loss_names[3])([frcnn_cls_actuals, frcnn_cls_predictions])
    #
    model = Model(inputs=[input_img, rpn_reg_actuals, rpn_cls_actuals, input_anchors, input_gt_boxes, input_gt_labels],
                  outputs=[rpn_reg_predictions, rpn_cls_predictions, frcnn_reg_predictions, frcnn_cls_predictions,
                           rpn_reg_loss_layer, rpn_cls_loss_layer, frcnn_reg_loss_layer, frcnn_cls_loss_layer])
    #
    for layer_name in loss_names:
        layer = model.get_layer(layer_name)
        model.add_loss(layer.output)
        model.add_metric(layer.output, name=layer_name, aggregation="mean")
    return model

def get_model_path(stride):
    main_path = "models"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "stride_{0}_frcnn_model_weights.h5".format(stride))
    return model_path
