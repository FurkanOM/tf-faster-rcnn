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

def generate_anchors(image_height, image_width, hyper_params):
    """Broadcasting base_anchors and generating all anchors for given image parameters.
    inputs:
        image_height = height of the image
        image_width = width of the image
        hyper_params = dictionary

    outputs:
        anchors = (output_width * output_height * anchor_count, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
    """
    anchor_count = hyper_params["anchor_count"]
    stride = hyper_params["stride"]
    output_height, output_width = image_height // stride, image_width // stride
    #
    grid_x = np.arange(0, output_width) * stride
    grid_y = np.arange(0, output_height) * stride
    #
    width_padding = (image_width - output_width * stride) / 2
    height_padding = (image_height - output_height * stride) / 2
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
    anchors = helpers.normalize_bboxes(anchors, image_height, image_width)
    anchors = np.clip(anchors, 0, 1)
    return anchors

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
            input_img, bbox_deltas, bbox_labels = get_step_data(image_data, anchors, hyper_params, input_processor)
            yield input_img, (bbox_deltas, bbox_labels)

def get_step_data(image_data, anchors, hyper_params, input_processor):
    """Generating one step data for training or inference.
    Batch operations supported.
    inputs:
        image_data =
            img (batch_size, height, width, channels)
            gt_boxes (batch_size, gt_box_size, [y1, x1, y2, x2])
                these values in normalized format between [0, 1]
            gt_labels (batch_size, gt_box_size)
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary
        input_processor = function for preparing image for input. It's getting from backbone.

    outputs:
        input_img = (batch_size, height, width, channels)
            preprocessed image using input_processor
        bbox_deltas = (batch_size, output_height, output_width, anchor_count * [delta_y, delta_x, delta_h, delta_w])
        bbox_labels = (batch_size, output_height, output_width, anchor_count)
    """
    img, gt_boxes, gt_labels = image_data
    batch_size, image_height, image_width = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2]
    input_img = input_processor(img)
    input_img = tf.image.convert_image_dtype(input_img, tf.float32)
    stride = hyper_params["stride"]
    anchor_count = hyper_params["anchor_count"]
    total_pos_bboxes = hyper_params["total_pos_bboxes"]
    total_neg_bboxes = hyper_params["total_neg_bboxes"]
    variances = hyper_params["variances"]
    output_height, output_width = image_height // stride, image_width // stride
    total_anchors = anchors.shape[0]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = helpers.generate_iou_map(anchors, gt_boxes)
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
    # Generate neg mask for neg bboxes
    neg_mask = tf.greater(sorted_map_indices, (total_anchors-1) - total_neg_bboxes)
    # Generate pos and negative labels
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    # Replace negative bboxes with zeros
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    # Calculate delta values between anchors and ground truth bboxes
    bbox_deltas = helpers.get_deltas_from_bboxes(anchors, expanded_gt_boxes) / variances
    #
    # bbox_deltas = tf.reshape(bbox_deltas, (batch_size, output_height, output_width, anchor_count * 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, output_height, output_width, anchor_count))
    #
    return input_img, bbox_deltas, bbox_labels

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
