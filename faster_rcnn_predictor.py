import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import helpers
import rpn
import faster_rcnn

args = helpers.handle_args()
if args.handle_gpu:
    helpers.handle_gpu_compatibility()

mode = "inference"
batch_size = 1
use_custom_images = False
custom_image_path = "data/images/"
hyper_params = helpers.get_hyper_params()

VOC_test_data, VOC_info = helpers.get_dataset("voc/2007", "test")
labels = helpers.get_labels(VOC_info)
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1
# If you want to use different dataset and don't know max height and width values
# You can use calculate_max_height_width method in helpers
max_height, max_width = helpers.VOC["max_height"], helpers.VOC["max_width"]

if use_custom_images:
    VOC_test_data = helpers.get_image_data_from_folder(custom_image_path, max_height, max_width)
else:
    VOC_test_data = VOC_test_data.map(lambda x : helpers.preprocessing(x, max_height, max_width))
    padded_shapes, padding_values = helpers.get_padded_batch_params()
    VOC_test_data = VOC_test_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

rpn_model, base_model = rpn.get_model(hyper_params)
anchors = rpn.generate_anchors(max_height, max_width, hyper_params)
frcnn_model = faster_rcnn.get_model(base_model, rpn_model, anchors, hyper_params, mode=mode)
#
frcnn_model_path = helpers.get_model_path("frcnn")
frcnn_model.load_weights(frcnn_model_path)

background_label = "bg"
labels = [background_label] + labels
bg_id = labels.index(background_label)
total_labels = hyper_params["total_labels"]

for image_data in VOC_test_data:
    img, _, _ = image_data
    input_img = preprocess_input(img)
    input_img = tf.image.convert_image_dtype(input_img, tf.float32)
    frcnn_pred = frcnn_model.predict_on_batch(input_img)
    roi_bboxes, rpn_reg_pred, rpn_cls_pred, frcnn_reg_pred, frcnn_cls_pred = frcnn_pred
    frcnn_reg_pred = tf.reshape(frcnn_reg_pred, (batch_size, tf.shape(frcnn_reg_pred)[1], total_labels, 4))
    frcnn_reg_pred *= hyper_params["variances"]
    #
    expanded_roi_bboxes = tf.tile(tf.expand_dims(roi_bboxes, -2), (1, 1, total_labels, 1))
    frcnn_bboxes = helpers.get_bboxes_from_deltas(expanded_roi_bboxes, frcnn_reg_pred)
    # We remove background predictions and reshape outputs for non max suppression
    frcnn_cls_pred = tf.cast(frcnn_cls_pred, tf.float32)
    pred_labels_map = tf.argmax(frcnn_cls_pred, 2, output_type=tf.int32)
    valid_cond = tf.not_equal(pred_labels_map, bg_id)
    #
    valid_bboxes = tf.expand_dims(frcnn_bboxes[valid_cond], 0)
    valid_labels = tf.expand_dims(frcnn_cls_pred[valid_cond], 0)
    #
    nms_bboxes, nmsed_scores, nmsed_classes, valid_detections = helpers.non_max_suppression(valid_bboxes, valid_labels,
                                                                                            max_output_size_per_class=10,
                                                                                            max_total_size=200, score_threshold=0.5)
    helpers.draw_bboxes_with_labels(img[0], nms_bboxes[0], nmsed_classes[0], nmsed_scores[0], labels)
