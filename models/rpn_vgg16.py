"""VGG16-backed Region Proposal Network model definition."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model


HyperParams = Dict[str, Any]


def get_model(hyper_params: HyperParams) -> Tuple[tf.keras.Model, tf.keras.layers.Layer]:
    """Build the VGG16 RPN model and expose its feature extractor layer.

    Args:
        hyper_params (Dict[str, Any]): Hyper-parameter dictionary containing image
            size and anchor count.

    Returns:
        Tuple[tf.keras.Model, tf.keras.layers.Layer]: The RPN model and the
        backbone feature extractor layer.
    """
    img_size = hyper_params["img_size"]
    base_model = VGG16(include_top=False, input_shape=(img_size, img_size, 3))
    feature_extractor = base_model.get_layer("block5_conv3")
    output = Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv")(feature_extractor.output)
    rpn_cls_output = Conv2D(hyper_params["anchor_count"], (1, 1), activation="sigmoid", name="rpn_cls")(output)
    rpn_reg_output = Conv2D(hyper_params["anchor_count"] * 4, (1, 1), activation="linear", name="rpn_reg")(output)
    rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
    return rpn_model, feature_extractor


def init_model(model: tf.keras.Model) -> None:
    """Warm up the model graph with dummy input before loading optimizer state.

    Args:
        model (tf.keras.Model): RPN model instance to initialize.

    Returns:
        None: The model graph is built in place.
    """
    model(tf.random.uniform((1, 500, 500, 3)))
