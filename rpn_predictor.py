import tensorflow as tf
from tensorflow.keras.models import load_model
import helpers
import rpn
import faster_rcnn

args = helpers.handle_args()
if args.handle_gpu:
    helpers.handle_gpu_compatibility()

batch_size = 1
# If you have trained faster rcnn model you can load weights from faster rcnn model
load_weights_from_frcnn = False
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

rpn_model, _ = rpn.get_model(hyper_params)

frcnn_model_path = helpers.get_model_path("frcnn")
rpn_model_path = helpers.get_model_path("rpn")
model_path = frcnn_model_path if load_weights_from_frcnn else rpn_model_path
rpn_model.load_weights(model_path, by_name=True)

anchors = rpn.generate_anchors(max_height, max_width, hyper_params)

for image_data in VOC_test_data:
    img, _, _ = image_data
    rpn_bbox_deltas, rpn_labels = rpn_model.predict_on_batch(img)
    #
    total_anchors = anchors.shape[0]
    rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, total_anchors, 4))
    rpn_bbox_deltas *= hyper_params["variances"]
    rpn_labels = tf.reshape(rpn_labels, (batch_size, total_anchors, 1))
    #
    rpn_bboxes = helpers.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)
    rpn_bboxes = tf.reshape(rpn_bboxes, (batch_size, total_anchors, 1, 4))
    #
    nms_bboxes, _, _, _ = helpers.non_max_suppression(rpn_bboxes, rpn_labels,
                                                max_output_size_per_class=hyper_params["test_nms_topn"],
                                                max_total_size=hyper_params["test_nms_topn"])
    helpers.draw_bboxes(img, nms_bboxes)
