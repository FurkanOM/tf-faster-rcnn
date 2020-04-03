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

mode = "inference"
batch_size = 1
hyper_params = helpers.get_hyper_params()

VOC_test_data, VOC_info = helpers.get_dataset("voc/2007", "test")
labels = helpers.get_labels(VOC_info)
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1
# If you want to use different dataset and don't know max height and width values
# You can use calculate_max_height_width method in helpers
max_height, max_width = helpers.VOC["max_height"], helpers.VOC["max_width"]
VOC_test_data = VOC_test_data.map(lambda x : helpers.preprocessing(x, max_height, max_width))

padded_shapes, padding_values = helpers.get_padded_batch_params()
VOC_test_data = VOC_test_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

base_model = VGG16(include_top=False)
if hyper_params["stride"] == 16:
    base_model = Sequential(base_model.layers[:-1])
rpn_model = rpn.get_model(base_model, hyper_params)
frcnn_model = faster_rcnn.get_model(base_model, rpn_model, hyper_params, mode=mode)
#
frcnn_model_path = helpers.get_model_path("frcnn", hyper_params["stride"])
frcnn_model.load_weights(frcnn_model_path)

for image_data in VOC_test_data:
    img, gt_boxes, gt_labels = image_data
    input_img, anchors = rpn.get_step_data(image_data, hyper_params, preprocess_input, mode=mode)
    frcnn_pred = frcnn_model.predict_on_batch([input_img, anchors, gt_boxes])
    roi_bboxes, rpn_reg_pred, rpn_cls_pred, frcnn_reg_pred, frcnn_cls_pred = frcnn_pred
    # We remove background predictions and reshape outputs for non max suppression
    valid_pred_bboxes, valid_pred_labels = faster_rcnn.get_valid_predictions(roi_bboxes, frcnn_reg_pred, frcnn_cls_pred, hyper_params["total_labels"])

    nms_bboxes, nmsed_scores, nmsed_classes, valid_detections = helpers.non_max_suppression(valid_pred_bboxes, valid_pred_labels,
                                                                                            max_output_size_per_class=3,
                                                                                            max_total_size=12, score_threshold=0.7)
    helpers.draw_bboxes_with_labels(img[0], nms_bboxes[0], nmsed_classes[0], nmsed_scores[0], labels)
