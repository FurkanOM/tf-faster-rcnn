import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np

def preprocessing(image_data, final_height, final_width, apply_augmentation=False):
    """Image resizing operation handled before batch operations.
    inputs:
        image_data = tensorflow dataset image_data
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, channels)
        gt_boxes = (gt_box_size, [y1, x1, y2, x2])
        gt_labels = (gt_box_size)
    """
    img = image_data["image"]
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if apply_augmentation:
        img, gt_boxes = randomly_apply_operation(flip_horizontally, img, gt_boxes)
    return img, gt_boxes, gt_labels

def get_random_bool():
    """Generating random boolean.
    outputs:
        random boolean 0d tensor
    """
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)

def randomly_apply_operation(operation, img, gt_boxes):
    """Randomly applying given method to image and ground truth boxes.
    inputs:
        operation = callable method
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_or_not_img = (final_height, final_width, depth)
        modified_or_not_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes),
        lambda: (img, gt_boxes)
    )

def flip_horizontally(img, gt_boxes):
    """Flip image horizontally and adjust the ground truth boxes.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_img = (height, width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack([gt_boxes[..., 0],
                                1.0 - gt_boxes[..., 3],
                                gt_boxes[..., 2],
                                1.0 - gt_boxes[..., 1]], -1)
    return flipped_img, flipped_gt_boxes

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

def get_image_data_from_folder(custom_image_path, final_height, final_width):
    """Generating image data like tensorflow dataset format for a given image path.
    This method could be used for custom image predictions.
    inputs:
        custom_image_path = folder of the custom images
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        image_data = (img, dummy_gt_boxes, dummy_gt_labels)
            img = (1, final_height, final_width, depth)
            dummy_gt_boxes = None
            dummy_gt_labels = None
    """
    image_data = []
    for path, dir, filenames in os.walk(custom_image_path):
        for filename in filenames:
            img_path = os.path.join(path, filename)
            image = Image.open(img_path)
            resized_image = image.resize((final_width, final_height), Image.LANCZOS)
            img = tf.expand_dims(np.array(resized_image), 0)
            img = tf.image.convert_image_dtype(img, tf.float32)
            image_data.append((img, None, None))
        break
    return image_data

def get_padded_batch_params():
    """Generating padded batch params for tensorflow datasets.
    outputs:
        padded_shapes = output shapes for (images, ground truth boxes, labels)
        padding_values = padding values with dtypes for (images, ground truth boxes, labels)
    """
    padded_shapes = ([None, None, None], [None, None], [None,])
    padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
    return padded_shapes, padding_values
