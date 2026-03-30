"""Command-line and filesystem helpers for training and inference scripts."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import tensorflow as tf


def get_log_path(model_type: str, backbone: str = "vgg16", custom_postfix: str = "") -> str:
    """Return the TensorBoard log directory for a model variant.

    Args:
        model_type (str): Model family name such as `"rpn"` or `"faster_rcnn"`.
        backbone (str): Backbone identifier used for the run.
        custom_postfix (str): Optional suffix appended to the directory name.

    Returns:
        str: TensorBoard log directory path.
    """
    return "logs/{}_{}{}/{}".format(model_type, backbone, custom_postfix, datetime.now().strftime("%Y%m%d-%H%M%S"))


def get_model_path(model_type: str, backbone: str = "vgg16") -> str:
    """Return the weight file path for the requested model and backbone.

    Args:
        model_type (str): Model family name such as `"rpn"` or `"faster_rcnn"`.
        backbone (str): Backbone identifier used for the run.

    Returns:
        str: Filesystem path to the weight file.
    """
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "{}_{}_model_weights.h5".format(model_type, backbone))
    return model_path


def handle_args() -> argparse.Namespace:
    """Parse command-line arguments shared by the training scripts.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Faster-RCNN Implementation")
    parser.add_argument("-handle-gpu", action="store_true", help="Tensorflow 2 GPU compatibility flag")
    parser.add_argument("--backbone", required=False,
                        default="mobilenet_v2",
                        metavar="['vgg16', 'mobilenet_v2']",
                        help="Which backbone used for the rpn")
    args = parser.parse_args()
    return args


def is_valid_backbone(backbone: str) -> None:
    """Validate that the selected backbone is supported.

    Args:
        backbone (str): Backbone name received from the command line.

    Returns:
        None: This function raises an `AssertionError` for unsupported values.
    """
    assert backbone in ["vgg16", "mobilenet_v2"]


def handle_gpu_compatibility() -> None:
    """Enable memory growth on detected GPUs when TensorFlow exposes them.

    Returns:
        None: GPU memory growth is configured in place when possible.
    """
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as error:
        print(error)
