import os
import argparse
from PIL import Image, ImageFont, ImageDraw
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
###############################################################
## Custom callback for model saving and early stopping
###############################################################
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_path, monitor, patience=0):
        super(CustomCallback, self).__init__()
        self.model_path = model_path
        self.monitor = monitor
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.patience_counter = 0
        self.best_loss = float("inf")
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.last_epoch = epoch
        current = logs.get(self.monitor)
        if np.less(current, self.best_loss):
            self.best_loss = current
            self.patience_counter = 0
            self.model.save_weights(self.model_path)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.last_epoch:
            print("Training early stopped at {0} epoch because loss value did not decrease last {1} epochs".format(self.last_epoch+1, self.patience))
###############################################################

def frcnn_cls_loss(*args):
    """Calculating faster rcnn class loss value.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )

    outputs:
        loss = CategoricalCrossentropy value
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    lf = tf.losses.CategoricalCrossentropy()
    return lf(y_true, y_pred)

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
    indices = tf.where(tf.not_equal(y_true, -1))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.BinaryCrossentropy()
    return lf(target, output)

def reg_loss(*args):
    """Calculating rpn/faster rcnn regression loss value.
    Reg value should be different than zero for actual values.
    Because of this we only take into account non zero values.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )

    outputs:
        loss = Huber it's almost the same with the smooth L1 loss
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    indices = tf.where(tf.not_equal(y_true, 0))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    # # Same with the smooth l1 loss
    lf = tf.losses.Huber()
    return lf(target, output)

def get_model_path(model_type, stride):
    """Generating model path from stride value for save/load model weights.
    inputs:
        model_type = "rpn" or "frcnn"
        stride = 32 or 16

    outputs:
        model_path = os model path, for example: "models/stride_32_rpn_model_weights.h5"
    """
    main_path = "models"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "stride_{0}_{1}_model_weights.h5".format(stride, model_type))
    return model_path

def get_hyper_params(**kwargs):
    """Generating hyper params in a dynamic way.
    inputs:
        **kwargs = any value could be updated in the hyper_params
            stride => should be 16 or 32
            nms_topn => should be <= (total_pos_bboxes + total_neg_bboxes) * 2

    outputs:
        hyper_params = dictionary
    """
    hyper_params = {
        "anchor_ratios": [0.5, 1, 2],
        "anchor_scales": [16, 32, 64, 128, 256],
        "stride": 32,
        "nms_topn": 300,
        "total_pos_bboxes": 64,
        "total_neg_bboxes": 64,
        "pooling_size": (7, 7),
    }
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value
    #
    hyper_params["anchor_count"] = len(hyper_params["anchor_ratios"]) * len(hyper_params["anchor_scales"])
    return hyper_params

def get_padded_batch_params():
    """Generating padded batch params for tensorflow datasets.
    outputs:
        padded_shapes = output shapes for (images, ground truth boxes, labels)
        padding_values = padding values with dtypes for (images, ground truth boxes, labels)
    """
    padded_shapes = ([None, None, None], [None, None], [None,])
    padding_values = (tf.constant(0, tf.uint8), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
    return padded_shapes, padding_values

def get_dataset(name, split, data_dir="~/tensorflow_datasets"):
    """Get tensorflow dataset split and info.
    inputs:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets

    outputs:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info

def get_step_size(total_items, batch_size):
    """Get step size for given total item size and batch size.
    inputs:
        total_items = number of total items
        batch_size = number of batch size during training or validation

    outputs:
        step_size = number of step size for model training
    """
    return int(np.ceil(total_items / batch_size))

def get_total_item_size(info, split):
    """Get total item size for given split.
    inputs:
        info = tensorflow dataset info
        split = data split string, should be one of ["train", "validation", "test"]

    outputs:
        total_item_size = number of total items
    """
    assert split in ["train", "train+validation", "validation", "test"]
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples

def get_labels(info):
    """Get label names list.
    inputs:
        info = tensorflow dataset info

    outputs:
        labels = [labels list]
    """
    return info.features["labels"].names

def preprocessing(image_data, max_height, max_width):
    """Image resizing operation handled before batch operations.
    inputs:
        image_data = tensorflow dataset image_data
        max_height = final image height after resizing
        max_width = final image width after resizing

    outputs:
        img = (max_height, max_width, channels)
        gt_boxes = (gt_box_size, [y1, x1, y2, x2])
        gt_labels = (gt_box_size)
    """
    img = image_data["image"]
    img = resize_image(img, max_height, max_width)
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)
    return img, gt_boxes, gt_labels

def get_image_params(batch_img, stride):
    """Generating image output width and height values using stride value.
    This method should be updated for backbones.
    It's only supporting VGG16 backbone for now.
    inputs:
        batch_img = (batch_size, height, width, channels)
        stride = 16 or 32 for now

    outputs:
        height = image height
        width = image width
        output_height = image output height for backbone
        output_width = image output width for backbone
    """
    img_shape = tf.shape(batch_img)
    height, width = img_shape[1], img_shape[2]
    output_height, output_width = height // stride, width // stride
    return height, width, output_height, output_width

def non_max_suppression(pred_bboxes, pred_labels, **kwargs):
    """Applying non maximum suppression.
    Details could be found on tensorflow documentation.
    https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    inputs:
        pred_bboxes = (batch_size, total_bboxes, total_labels, [y1, x1, y2, x2])
            total_labels should be 1 for binary operations like in rpn
        pred_labels = (batch_size, total_bboxes, total_labels)
        **kwargs = other parameters

    outputs:
        nms_boxes = (batch_size, max_detections, [y1, x1, y2, x2])
        nmsed_scores = (batch_size, max_detections)
        nmsed_classes = (batch_size, max_detections)
        valid_detections = (batch_size)
            Only the top valid_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid.
            The rest of the entries are zero paddings.
    """
    return tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        **kwargs
    )

def get_bboxes_from_deltas(anchors, deltas):
    """Calculating bounding boxes for given bounding box and delta values.
    inputs:
        anchors = (batch_size, total_bboxes, [y1, x1, y2, x2])
        deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])

    outputs:
        final_boxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[..., 0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height
    #
    all_bbox_width = tf.exp(deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_anc_height) + all_anc_ctr_y
    #
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    #
    return tf.stack([y1, x1, y2, x2], axis=2)

def get_deltas_from_bboxes(bboxes, gt_boxes):
    """Calculating bounding box deltas for given bounding box and ground truth boxes.
    inputs:
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
        gt_boxes = (batch_size, total_bboxes, [y1, x1, y2, x2])

    outputs:
        final_deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
    """
    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height
    #
    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height
    #
    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
    #
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)

def generate_iou_map(bboxes, gt_boxes):
    """Calculating iou values for each ground truth boxes in batched manner.
    inputs:
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
        gt_boxes = (batch_size, total_gt_boxes, [y1, x1, y2, x2])

    outputs:
        iou_map = (batch_size, total_bboxes, total_gt_boxes)
    """
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=2)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=2)
    # Calculate bbox and ground truth boxes areas
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=2)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=2)
    #
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    ### Calculate intersection area
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    ### Calculate union area
    union_area = (tf.expand_dims(bbox_area, 2) + tf.expand_dims(gt_area, 1) - intersection_area)
    # Intersection over Union
    return intersection_area / union_area

def normalize_bboxes(bboxes, height, width):
    """Normalizing bounding boxes.
    inputs:
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
        height = image height
        width = image width

    outputs:
        normalized_bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    new_bboxes = np.zeros(bboxes.shape, dtype=np.float32)
    new_bboxes[:, 0] = bboxes[:, 0] / height
    new_bboxes[:, 1] = bboxes[:, 1] / width
    new_bboxes[:, 2] = bboxes[:, 2] / height
    new_bboxes[:, 3] = bboxes[:, 3] / width
    return new_bboxes

def denormalize_bboxes(bboxes, height, width):
    """Denormalizing bounding boxes.
    inputs:
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
        height = image height
        width = image width

    outputs:
        denormalized_bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    new_bboxes = np.zeros(bboxes.shape, dtype=np.float32)
    new_bboxes[:, 0] = np.round(bboxes[:, 0] * height)
    new_bboxes[:, 1] = np.round(bboxes[:, 1] * width)
    new_bboxes[:, 2] = np.round(bboxes[:, 2] * height)
    new_bboxes[:, 3] = np.round(bboxes[:, 3] * width)
    return new_bboxes

def resize_image(img, final_height, final_width):
    """Resize image to given height and width values.
    inputs:
        img = (height, width, channels)
        final_height = final image height after resizing
        final_width = final image width after resizing

    outputs:
        resized_img = (final_height, final_width, channels)
    """
    resized_img = tf.image.resize(tf.image.convert_image_dtype(img, tf.float32), (final_height, final_width))
    return tf.image.convert_image_dtype(resized_img, tf.uint8)

def img_from_array(array):
    """Getting pillow image object from numpy array.
    inputs:
        array = (height, width, channels)

    outputs:
        image = Pillow image object
    """
    return Image.fromarray(array)

def array_from_img(image):
    """Getting numpy array from pillow image object.
    inputs:
        image = Pillow image object

    outputs:
        array = (height, width, channels)
    """
    return np.array(image)

def draw_grid_map(img, grid_map, stride):
    """Drawing grid intersection on given image.
    inputs:
        img = (height, width, channels)
        grid_map = (output_height * output_width, [y_index, x_index, y_index, x_index])
            tiled x, y coordinates
        stride = number of stride

    outputs:
        array = (height, width, channels)
    """
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
    """Drawing bounding boxes on given image.
    inputs:
        img = (batch_size, height, width, channels)
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    colors = tf.cast(np.array([[1, 0, 0, 1]] * 10), dtype=tf.float32)
    img_with_bounding_boxes = tf.image.draw_bounding_boxes(
        img,
        bboxes,
        colors
    )
    plt.figure()
    plt.imshow(img_with_bounding_boxes[0])
    plt.show()

def draw_bboxes_with_labels(img, bboxes, label_indices, probs, labels):
    """Drawing bounding boxes with labels on given image.
    inputs:
        img = (height, width, channels)
        bboxes = (total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
        label_indices = (total_bboxes)
        probs = (total_bboxes)
        labels = [labels string list]
    """
    colors = []
    for i in range(len(labels)):
        colors.append(tuple(np.random.choice(range(256), size=4)))
    image = tf.keras.preprocessing.image.array_to_img(img)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    denormalized_bboxes = denormalize_bboxes(bboxes, height, width)
    for index, bbox in enumerate(denormalized_bboxes):
        y1, x1, y2, x2 = np.split(bbox, 4)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        label_index = int(label_indices[index])
        color = colors[label_index]
        label_text = "{0} {1:0.3f}".format(labels[label_index], probs[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    #
    plt.figure()
    plt.imshow(image)
    plt.show()

def calculate_max_height_width(imgs):
    """Calculating max height and max width values for given imgs.
    inputs:
        imgs = (batch_size, height, width, channels)

    outputs:
        max_height = maximum height value of all images
        max_width = maximum width value of all images
    """
    h_w_map = np.zeros((len(imgs), 2), dtype=np.int32)
    for index, img in enumerate(imgs):
        h_w_map[index, 0], h_w_map[index, 1], _ = img.shape
    max_val = h_w_map.argmax(axis=0)
    max_height, max_width = h_w_map[max_val[0], 0], h_w_map[max_val[1], 1]
    return max_height, max_width

def handle_args():
    """Handling of command line arguments using argparse library.

    outputs:
        args = parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Faster-RCNN Implementation")
    parser.add_argument("-handle-gpu", action="store_true", help="Tensorflow 2 GPU compatibility flag")
    args = parser.parse_args()
    return args

def handle_gpu_compatibility():
    """Handling of GPU issues for cuDNN initialize error and memory issues."""
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)
