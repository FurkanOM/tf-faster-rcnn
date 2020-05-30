import tensorflow as tf
import math
from utils import bbox_utils

RPN = {
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

def get_hyper_params(backbone, **kwargs):
    """Generating hyper params in a dynamic way.
    inputs:
        **kwargs = any value could be updated in the hyper_params

    outputs:
        hyper_params = dictionary
    """
    hyper_params = RPN[backbone]
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
    #
    hyper_params["anchor_count"] = len(hyper_params["anchor_ratios"]) * len(hyper_params["anchor_scales"])
    return hyper_params

def get_step_size(total_items, batch_size):
    """Get step size for given total item size and batch size.
    inputs:
        total_items = number of total items
        batch_size = number of batch size during training or validation
    outputs:
        step_size = number of step size for model training
    """
    return math.ceil(total_items / batch_size)

def randomly_select_xyz_mask(mask, select_xyz):
    """Selecting x, y, z number of True elements for corresponding batch and replacing others to False
    inputs:
        mask = (batch_size, [m_bool_value])
        select_xyz = ([x_y_z_number_for_corresponding_batch])
            example = tf.constant([128, 50, 42], dtype=tf.int32)
    outputs:
        selected_valid_mask = (batch_size, [m_bool_value])
    """
    maxval = tf.reduce_max(select_xyz) * 10
    random_mask = tf.random.uniform(tf.shape(mask), minval=1, maxval=maxval, dtype=tf.int32)
    multiplied_mask = tf.cast(mask, tf.int32) * random_mask
    sorted_mask = tf.argsort(multiplied_mask, direction="DESCENDING")
    sorted_mask_indices = tf.argsort(sorted_mask)
    selected_mask = tf.less(sorted_mask_indices, tf.expand_dims(select_xyz, 1))
    return tf.logical_and(mask, selected_mask)

def faster_rcnn_generator(dataset, anchors, hyper_params):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params)
            yield (img, gt_boxes, gt_labels, bbox_deltas, bbox_labels), ()

def rpn_generator(dataset, anchors, hyper_params):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params)
            yield img, (bbox_deltas, bbox_labels)

def calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params):
    """Generating one step data for training or inference.
    Batch operations supported.
    inputs:
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_boxes (batch_size, gt_box_size, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_labels (batch_size, gt_box_size)
        hyper_params = dictionary

    outputs:
        bbox_deltas = (batch_size, total_anchors, [delta_y, delta_x, delta_h, delta_w])
        bbox_labels = (batch_size, feature_map_shape, feature_map_shape, anchor_count)
    """
    batch_size = tf.shape(gt_boxes)[0]
    feature_map_shape = hyper_params["feature_map_shape"]
    anchor_count = hyper_params["anchor_count"]
    total_pos_bboxes = hyper_params["total_pos_bboxes"]
    total_neg_bboxes = hyper_params["total_neg_bboxes"]
    variances = hyper_params["variances"]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = bbox_utils.generate_iou_map(anchors, gt_boxes)
    # Get max index value for each row
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # Get max index value for each column
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_mask = tf.greater(merged_iou_map, 0.7)
    #
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    #
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    max_pos_mask = tf.scatter_nd(scatter_bbox_indices, tf.fill((tf.shape(valid_indices)[0], ), True), tf.shape(pos_mask))
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    pos_mask = randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
    #
    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count
    #
    neg_mask = tf.logical_and(tf.less(merged_iou_map, 0.3), tf.logical_not(pos_mask))
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count)
    #
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_row, batch_dims=1)
    # Replace negative bboxes with zeros
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    # Calculate delta values between anchors and ground truth bboxes
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(anchors, expanded_gt_boxes) / variances
    #
    # bbox_deltas = tf.reshape(bbox_deltas, (batch_size, feature_map_shape, feature_map_shape, anchor_count * 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, feature_map_shape, feature_map_shape, anchor_count))
    #
    return bbox_deltas, bbox_labels

def frcnn_cls_loss(*args):
    """Calculating faster rcnn class loss value.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )

    outputs:
        loss = CategoricalCrossentropy value
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    loss_fn = tf.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
    loss_for_all = loss_fn(y_true, y_pred)
    #
    cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    mask = tf.cast(cond, dtype=tf.float32)
    #
    conf_loss = tf.reduce_sum(mask * loss_for_all)
    total_boxes = tf.maximum(1.0, tf.reduce_sum(mask))
    return conf_loss / total_boxes

def rpn_cls_loss(*args):
    """Calculating rpn class loss value.
    Rpn actual class value should be 0 or 1.
    Because of this we only take into account non -1 values.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )

    outputs:
        loss = BinaryCrossentropy value
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    indices = tf.where(tf.not_equal(y_true, tf.constant(-1.0, dtype=tf.float32)))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.BinaryCrossentropy()
    return lf(target, output)

def reg_loss(*args):
    """Calculating rpn / faster rcnn regression loss value.
    Reg value should be different than zero for actual values.
    Because of this we only take into account non zero values.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )

    outputs:
        loss = Huber it's almost the same with the smooth L1 loss
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, 4))
    #
    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    loss_for_all = loss_fn(y_true, y_pred)
    loss_for_all = tf.reduce_sum(loss_for_all, axis=-1)
    #
    pos_cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    #
    loc_loss = tf.reduce_sum(pos_mask * loss_for_all)
    total_pos_bboxes = tf.maximum(1.0, tf.reduce_sum(pos_mask))
    return loc_loss / total_pos_bboxes
