import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential

def get_model(hyper_params):
    """Generating rpn model for given hyper params.
    inputs:
        hyper_params = dictionary

    outputs:
        rpn_model = tf.keras.model
    """
    base_model = VGG16(include_top=False)
    base_model = Sequential(base_model.layers[:-1])
    output = Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv")(base_model.output)
    rpn_cls_output = Conv2D(hyper_params["anchor_count"], (1, 1), activation="sigmoid", name="rpn_cls")(output)
    rpn_reg_output = Conv2D(hyper_params["anchor_count"] * 4, (1, 1), activation="linear", name="rpn_reg")(output)
    rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
    return rpn_model, base_model

def init_model(model):
    """Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model
    """
    model(tf.random.uniform((1, 512, 512, 3)))
