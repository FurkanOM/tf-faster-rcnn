import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import helpers
import rpn
import faster_rcnn

args = helpers.handle_args()
if args.handle_gpu:
    helpers.handle_gpu_compatibility()

batch_size = 4
epochs = 50
load_weights = False
hyper_params = helpers.get_hyper_params()

VOC_train_data, VOC_info = helpers.get_dataset("voc/2007", "train+validation")
VOC_val_data, _ = helpers.get_dataset("voc/2007", "test")
VOC_train_total_items = helpers.get_total_item_size(VOC_info, "train+validation")
VOC_val_total_items = helpers.get_total_item_size(VOC_info, "test")
step_size_train = helpers.get_step_size(VOC_train_total_items, batch_size)
step_size_val = helpers.get_step_size(VOC_val_total_items, batch_size)
labels = helpers.get_labels(VOC_info)
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1
# If you want to use different dataset and don't know max height and width values
# You can use calculate_max_height_width method in helpers
max_height, max_width = helpers.VOC["max_height"], helpers.VOC["max_width"]
VOC_train_data = VOC_train_data.map(lambda x : helpers.preprocessing(x, max_height, max_width, apply_augmentation=True))
VOC_val_data = VOC_val_data.map(lambda x : helpers.preprocessing(x, max_height, max_width))

padded_shapes, padding_values = helpers.get_padded_batch_params()
VOC_train_data = VOC_train_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
VOC_val_data = VOC_val_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

anchors = rpn.generate_anchors(max_height, max_width, hyper_params)
frcnn_train_feed = faster_rcnn.generator(VOC_train_data, anchors, hyper_params)
frcnn_val_feed = faster_rcnn.generator(VOC_val_data, anchors, hyper_params)
#
rpn_model, base_model = rpn.get_model(hyper_params)
frcnn_model = faster_rcnn.get_model(base_model, rpn_model, anchors, hyper_params)
frcnn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5),
                    loss=[None] * len(frcnn_model.output))
faster_rcnn.init_model(frcnn_model, hyper_params)
# If you have pretrained rpn model
# You can load rpn weights for faster training
rpn_load_weights = False
if rpn_load_weights:
    rpn_model_path = helpers.get_model_path("rpn")
    rpn_model.load_weights(rpn_model_path)
# Load weights
frcnn_model_path = helpers.get_model_path("frcnn")

if load_weights:
    frcnn_model.load_weights(frcnn_model_path)

checkpoint_callback = ModelCheckpoint(frcnn_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)

frcnn_model.fit(frcnn_train_feed,
                steps_per_epoch=step_size_train,
                validation_data=frcnn_val_feed,
                validation_steps=step_size_val,
                epochs=epochs,
                callbacks=[checkpoint_callback])
