import os
import argparse
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

###########################################
## Pascal VOC
###########################################
VOC = {
    "max_height": 500,
    "max_width": 500,
}

def get_VOC_data(split):
    assert split in ["train", "validation", "test"]
    dataset, info = tfds.load("voc", split=split, with_info=True)
    total_labels = info.features["labels"].num_classes
    data_len = info.splits[split].num_examples
    return dataset, data_len, total_labels

def preprocessing(image_data, max_height, max_width):
    img = image_data["image"]
    img_shape = tf.shape(img)
    padding = get_padding(img_shape[0], img_shape[1], max_height, max_width)
    gt_boxes = update_gt_boxes(image_data["objects"]["bbox"], img_shape[0], img_shape[1], padding)
    img = get_padded_img(img, max_height, max_width)
    img_float32 = tf.image.convert_image_dtype(img, dtype=tf.float32)
    gt_labels = tf.cast(image_data["objects"]["label"], tf.int32)
    return img_float32, gt_boxes, gt_labels

def get_image_params(batch_img, stride):
    img_shape = tf.shape(batch_img)
    height, width = img_shape[1], img_shape[2]
    output_height, output_width = height // stride, width // stride
    return height, width, output_height, output_width

def generate_iou_map(bboxes, gt_boxes):
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=1)
    # Calculate bbox and ground truth boxes areas
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=1)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=1)
    #
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2))
    ### Calculate intersection area
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    ### Calculate union area
    union_area = (tf.expand_dims(bbox_area, 1) + tf.expand_dims(gt_area, 0) - intersection_area)
    # Intersection over Union
    return intersection_area / union_area

def get_deltas_from_bboxes(bboxes, gt_boxes):
    bbox_width = bboxes[:, :, 3] - bboxes[:, :, 1]
    bbox_height = bboxes[:, :, 2] - bboxes[:, :, 0]
    bbox_ctr_x = bboxes[:, :, 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[:, :, 0] + 0.5 * bbox_height
    #
    gt_width = gt_boxes[:, :, 3] - gt_boxes[:, :, 1]
    gt_height = gt_boxes[:, :, 2] - gt_boxes[:, :, 0]
    gt_ctr_x = gt_boxes[:, :, 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[:, :, 0] + 0.5 * gt_height
    #
    bbox_width = tf.where(tf.equal(bbox_width, 0), tf.ones_like(bbox_width), bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), tf.ones_like(bbox_height), bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(bbox_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(bbox_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(bbox_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(bbox_height), tf.math.log(gt_height / bbox_height))
    #
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=2)

def get_selected_indices(args):
    bboxes, gt_boxes, total_pos_bboxes, total_neg_bboxes = args
    # remove paddings coming from batch operation
    cond = tf.reduce_any(tf.not_equal(gt_boxes, -1), axis=1)
    gt_boxes_without_pad = tf.boolean_mask(gt_boxes, cond)
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = generate_iou_map(bboxes, gt_boxes_without_pad)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=1)
    # Sorted iou values
    sorted_iou_map = tf.argsort(merged_iou_map, direction="DESCENDING")
    # Get highest and lowest candidate indices
    pos_candidate_indices = sorted_iou_map[:total_pos_bboxes * 2]
    neg_candidate_indices = sorted_iou_map[::-1][:total_neg_bboxes * 2]
    # Randomly select pos and neg indices from candidates
    pos_bbox_indices = tf.random.shuffle(pos_candidate_indices)[:total_pos_bboxes]
    neg_bbox_indices = tf.random.shuffle(neg_candidate_indices)[:total_neg_bboxes]
    gt_box_indices = tf.gather(max_indices_each_gt_box, pos_bbox_indices)
    #
    bbox_indices = tf.concat([pos_bbox_indices, neg_bbox_indices], 0)
    return bbox_indices, gt_box_indices

def get_tiled_indices(batch_size, row_size):
    tiled_indices = tf.range(batch_size)
    tiled_indices = tf.tile(tf.expand_dims(tiled_indices, axis=1), (1, row_size))
    tiled_indices = tf.reshape(tiled_indices, (-1, 1))
    return tiled_indices

def get_gt_boxes_map(gt_boxes, gt_box_indices, batch_size, total_neg_bboxes):
    pos_gt_boxes_map = tf.gather(gt_boxes, gt_box_indices, batch_dims=1)
    neg_gt_boxes_map = tf.zeros((batch_size, total_neg_bboxes, 4), tf.float32)
    return tf.concat([pos_gt_boxes_map, neg_gt_boxes_map], axis=1)

def get_scatter_indices_for_bboxes(flatted_indices, batch_size, total_bboxes):
    indices_size = len(flatted_indices)
    scatter_indices = tf.concat(flatted_indices, 1)
    return tf.reshape(scatter_indices, (batch_size, total_bboxes, indices_size))

def normalize_bboxes(bboxes, height, width):
    new_bboxes = np.zeros(bboxes.shape, dtype=np.float32)
    new_bboxes[:, 0] = bboxes[:, 0] / height
    new_bboxes[:, 1] = bboxes[:, 1] / width
    new_bboxes[:, 2] = bboxes[:, 2] / height
    new_bboxes[:, 3] = bboxes[:, 3] / width
    return new_bboxes

def update_gt_boxes(gt_boxes, img_height, img_width, padding):
    img_height = tf.cast(img_height, tf.float32)
    img_width = tf.cast(img_width, tf.float32)
    padded_height = img_height + padding[0] + padding[2]
    padded_width = img_width + padding[1] + padding[3]
    y1 = (tf.round(gt_boxes[:, 0] * img_height) + padding[0]) / padded_height
    x1 = (tf.round(gt_boxes[:, 1] * img_width) + padding[1]) / padded_width
    y2 = (tf.round(gt_boxes[:, 2] * img_height) + padding[0]) / padded_height
    x2 = (tf.round(gt_boxes[:, 3] * img_width) + padding[1]) / padded_width
    return tf.stack([y1, x1, y2, x2], axis=1)

def get_padded_img(img, max_height, max_width):
    return tf.image.resize_with_crop_or_pad(
        img,
        max_height,
        max_width
    )

def get_padding(img_height, img_width, max_height, max_width):
    padding_height = max_height - img_height
    padding_width = max_width - img_width
    top = padding_height // 2
    bottom = padding_height - top
    left = padding_width // 2
    right = padding_width - left
    return tf.cast(tf.stack([top, left, bottom, right]), tf.float32)

def img_from_array(array):
    return Image.fromarray(array)

def array_from_img(image):
    return np.array(image)

def draw_grid_map(img, grid_map, stride):
    image = img_from_array(img)
    draw = ImageDraw.Draw(image)
    counter = 0
    for grid in grid_map:
        draw.rectangle((
            grid[0] + stride // 2 - 2,
            grid[1] + stride // 2 - 2,
            grid[2] + stride // 2 + 2,
            grid[3] + stride // 2 + 2), fill=(255, 255, 255, 0))
        counter += 1
    plt.figure()
    plt.imshow(image)
    plt.show()

def draw_bboxes(img, bboxes):
    colors = tf.cast(np.array([[1, 0, 0, 1]] * 10), dtype=tf.float32)
    img_with_bounding_boxes = tf.image.draw_bounding_boxes(
        img,
        bboxes,
        colors
    )
    plt.figure()
    plt.imshow(img_with_bounding_boxes[0])
    plt.show()

# It take images as numpy arrays and return max height, max width values
def calculate_max_height_width(imgs):
    h_w_map = np.zeros((len(imgs), 2), dtype=np.int32)
    for index, img in enumerate(imgs):
        h_w_map[index, 0], h_w_map[index, 1], _ = img.shape
    max_val = h_w_map.argmax(axis=0)
    max_height, max_width = h_w_map[max_val[0], 0], h_w_map[max_val[1], 1]
    return max_height, max_width

def handle_args():
    parser = argparse.ArgumentParser(description="Region Proposal Network Implementation")
    parser.add_argument("-handle-gpu", action="store_true", help="Tensorflow 2 GPU compatibility flag")
    args = parser.parse_args()
    return args

def handle_gpu_compatibility():
    # For tf2 GPU compatibility
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)
