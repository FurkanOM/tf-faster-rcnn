import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
import numpy as np
import helpers

def generate_base_anchors(hyper_params):
    """Generating top left anchors for given anchor_ratios, anchor_scales and stride values.
    inputs:
        hyper_params = dictionary

    outputs:
        base_anchors = (anchor_count, [y1, x1, y2, x2])
            these values not normalized yet
    """
    stride = hyper_params["stride"]
    anchor_ratios = hyper_params["anchor_ratios"]
    anchor_scales = hyper_params["anchor_scales"]
    center = stride // 2
    base_anchors = []
    for scale in anchor_scales:
        for ratio in anchor_ratios:
            box_area = scale ** 2
            w = round((box_area / ratio) ** 0.5)
            h = round(w * ratio)
            x_min = center - w / 2
            y_min = center - h / 2
            x_max = center + w / 2
            y_max = center + h / 2
            base_anchors.append([y_min, x_min, y_max, x_max])
    return np.array(base_anchors, dtype=np.float32)

def generate_anchors(img_params, hyper_params):
    """Broadcasting base_anchors and generating all anchors for given image parameters.
    inputs:
        img_params = (image height, image width, image output height, image output width)
            these output values need to be calculated for used backbone,
            for VGG16 output dimensions = dimension (height or width) // stride
        hyper_params = dictionary

    outputs:
        anchors = (output_width * output_height * anchor_count, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
    """
    anchor_count = hyper_params["anchor_count"]
    stride = hyper_params["stride"]
    height, width, output_height, output_width = img_params
    #
    grid_x = np.arange(0, output_width) * stride
    grid_y = np.arange(0, output_height) * stride
    #
    width_padding = (width - output_width * stride) / 2
    height_padding = (height - output_height * stride) / 2
    grid_x = width_padding + grid_x
    grid_y = height_padding + grid_y
    #
    grid_y, grid_x = np.meshgrid(grid_y, grid_x)
    grid_map = np.vstack((grid_y.ravel(), grid_x.ravel(), grid_y.ravel(), grid_x.ravel())).transpose()
    #
    base_anchors = generate_base_anchors(hyper_params)
    #
    output_area = grid_map.shape[0]
    anchors = base_anchors.reshape((1, anchor_count, 4)) + \
              grid_map.reshape((1, output_area, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((output_area * anchor_count, 4)).astype(np.float32)
    anchors = helpers.normalize_bboxes(anchors, height, width)
    anchors = np.clip(anchors, 0, 1)
    return anchors

def generator(dataset, hyper_params, input_processor):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        hyper_params = dictionary
        input_processor = function for preparing image for input. It's getting from backbone.

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            input_img, bbox_deltas, bbox_labels, _ = get_step_data(image_data, hyper_params, input_processor)
            yield input_img, (bbox_deltas, bbox_labels)

def get_step_data(image_data, hyper_params, input_processor, mode="training"):
    """Generating one step data for training or inference.
    Batch operations supported.
    inputs:
        image_data =
            img (batch_size, height, width, channels)
            gt_boxes (batch_size, gt_box_size, [y1, x1, y2, x2])
                these values in normalized format between [0, 1]
            gt_labels (batch_size, gt_box_size)
        hyper_params = dictionary
        input_processor = function for preparing image for input. It's getting from backbone.
        mode = "training" or "inference"

    outputs:
        input_img = (batch_size, height, width, channels)
            preprocessed image using input_processor
        bbox_deltas = (batch_size, output_height, output_width, anchor_count * [delta_y, delta_x, delta_h, delta_w])
            actual outputs for rpn, generating only training mode
        bbox_labels = (batch_size, output_height, output_width, anchor_count)
            actual outputs for rpn, generating only training mode
        anchors = (batch_size, output_height * output_width * anchor_count, [y1, x1, y2, x2])
    """
    img, gt_boxes, gt_labels = image_data
    batch_size = tf.shape(img)[0]
    input_img = input_processor(img)
    stride = hyper_params["stride"]
    anchor_count = hyper_params["anchor_count"]
    total_pos_bboxes = hyper_params["total_pos_bboxes"]
    total_neg_bboxes = hyper_params["total_neg_bboxes"]
    total_bboxes = total_pos_bboxes + total_neg_bboxes
    img_params = helpers.get_image_params(img, stride)
    height, width, output_height, output_width = img_params
    total_anchors = output_height * output_width * anchor_count
    anchors = generate_anchors(img_params, hyper_params)
    # We use same anchors for each batch so we multiplied anchors to the batch size
    anchors = tf.reshape(tf.tile(anchors, (batch_size, 1)), (batch_size, total_anchors, 4))
    if mode != "training":
        return input_img, anchors
    ################################################################################################################
    pos_bbox_indices, neg_bbox_indices, gt_box_indices = helpers.get_selected_indices(anchors, gt_boxes, total_pos_bboxes, total_neg_bboxes)
    #
    gt_boxes_map = helpers.get_gt_boxes_map(gt_boxes, gt_box_indices, batch_size, total_neg_bboxes)
    #
    pos_labels_map = tf.ones((batch_size, total_pos_bboxes), tf.int32)
    neg_labels_map = tf.zeros((batch_size, total_neg_bboxes), tf.int32)
    gt_labels_map = tf.concat([pos_labels_map, neg_labels_map], axis=1)
    #
    bbox_indices = tf.concat([pos_bbox_indices, neg_bbox_indices], axis=1)
    #
    flatted_batch_indices = helpers.get_tiled_indices(batch_size, total_bboxes)
    flatted_bbox_indices = tf.reshape(bbox_indices, (-1, 1))
    scatter_indices = helpers.get_scatter_indices_for_bboxes([flatted_batch_indices, flatted_bbox_indices], batch_size, total_bboxes)
    expanded_gt_boxes = tf.scatter_nd(scatter_indices, gt_boxes_map, tf.shape(anchors))
    #
    bbox_deltas = helpers.get_deltas_from_bboxes(anchors, expanded_gt_boxes)
    #
    bbox_labels = tf.negative(tf.ones((batch_size, total_anchors), tf.int32))
    bbox_labels = tf.tensor_scatter_nd_update(bbox_labels, scatter_indices, gt_labels_map)
    #
    bbox_deltas = tf.reshape(bbox_deltas, (batch_size, output_height, output_width, anchor_count * 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, output_height, output_width, anchor_count))
    #
    return input_img, bbox_deltas, bbox_labels, anchors

def get_model(base_model, hyper_params):
    """Generating rpn model for given backbone base model and hyper params.
    inputs:
        base_model = tf.keras.model pretrained backbone, only VGG16 available for now
        hyper_params = dictionary

    outputs:
        rpn_model = tf.keras.model
    """
    output = Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv")(base_model.output)
    rpn_cls_output = Conv2D(hyper_params["anchor_count"], (1, 1), activation="sigmoid", name="rpn_cls")(output)
    rpn_reg_output = Conv2D(hyper_params["anchor_count"] * 4, (1, 1), activation="linear", name="rpn_reg")(output)
    rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
    return rpn_model
