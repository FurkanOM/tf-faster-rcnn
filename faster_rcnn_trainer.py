import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import helpers
import rpn
import faster_rcnn

args = helpers.handle_args()
if args.handle_gpu:
    helpers.handle_gpu_compatibility()

train_batch_size = 2
val_batch_size = 8
epochs = 50
rpn_load_weights = False
frcnn_load_weights = False
hyper_params = helpers.get_hyper_params()

VOC_train_data, VOC_info = helpers.get_VOC_data("train")
VOC_val_data, _ = helpers.get_VOC_data("validation")
VOC_train_total_items = helpers.get_total_item_size(VOC_info, "train")
VOC_val_total_items = helpers.get_total_item_size(VOC_info, "validation")
labels = helpers.get_labels(VOC_info)
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1

rpn_model_path = rpn.get_model_path(hyper_params["stride"])
frcnn_model_path = faster_rcnn.get_model_path(hyper_params["stride"])

base_model = VGG16(include_top=False)
if hyper_params["stride"] == 16:
    base_model = Sequential(base_model.layers[:-1])
#
rpn_model = rpn.get_model(base_model, hyper_params)
if rpn_load_weights:
    rpn_model.load_weights(rpn_model_path)
#
frcnn_model = faster_rcnn.get_model(base_model, rpn_model, hyper_params)
if frcnn_load_weights:
    frcnn_model.load_weights(frcnn_model_path)

frcnn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5),
                    loss=[None] * len(frcnn_model.output))

# If you want to use different dataset and don't know max height and width values
# You can use calculate_max_height_width method in helpers
max_height, max_width = helpers.VOC["max_height"], helpers.VOC["max_width"]
VOC_train_data = VOC_train_data.map(lambda x : helpers.preprocessing(x, max_height, max_width))
VOC_val_data = VOC_val_data.map(lambda x : helpers.preprocessing(x, max_height, max_width))

padded_shapes, padding_values = helpers.get_padded_batch_params()
VOC_train_data = VOC_train_data.padded_batch(train_batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
VOC_val_data = VOC_val_data.padded_batch(val_batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

frcnn_train_feed = faster_rcnn.generator(VOC_train_data, hyper_params, preprocess_input)
frcnn_val_feed = faster_rcnn.generator(VOC_val_data, hyper_params, preprocess_input)

model_checkpoint = ModelCheckpoint(frcnn_model_path, save_best_only=True, save_weights_only=True, monitor="val_loss", mode="auto")
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=0, mode="auto")

step_size_train = VOC_train_total_items // train_batch_size
step_size_val = VOC_val_total_items // val_batch_size
frcnn_model.fit(frcnn_train_feed,
                steps_per_epoch=step_size_train,
                validation_data=frcnn_val_feed,
                validation_steps=step_size_val,
                epochs=epochs,
                callbacks=[early_stopping, model_checkpoint])
