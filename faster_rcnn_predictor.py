import tensorflow as tf
from utils import io_utils, data_utils, train_utils, bbox_utils, drawing_utils
from models import faster_rcnn

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

mode = "inference"
batch_size = 1
use_custom_images = False
custom_image_path = "data/images/"
backbone = args.backbone
io_utils.is_valid_backbone(backbone)

if backbone == "mobilenet_v2":
    from models.rpn_mobilenet_v2 import get_model as get_rpn_model
else:
    from models.rpn_vgg16 import get_model as get_rpn_model

hyper_params = train_utils.get_hyper_params(backbone)

test_data, dataset_info = data_utils.get_dataset("voc/2007", "test")
labels = data_utils.get_labels(dataset_info)
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1
#
img_size = hyper_params["img_size"]

if use_custom_images:
    test_data = data_utils.get_image_data_from_folder(custom_image_path, img_size, img_size)
else:
    test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))
    padded_shapes, padding_values = data_utils.get_padded_batch_params()
    test_data = test_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
#
anchors = bbox_utils.generate_anchors(hyper_params)
rpn_model, feature_extractor = get_rpn_model(hyper_params)
frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params, mode=mode)
#
frcnn_model_path = io_utils.get_model_path("faster_rcnn", backbone)
frcnn_model.load_weights(frcnn_model_path)

background_label = "bg"
labels = [background_label] + labels
bg_id = labels.index(background_label)
total_labels = hyper_params["total_labels"]

for image_data in test_data:
    img, _, _ = image_data
    frcnn_pred = frcnn_model.predict_on_batch(img)
    roi_bboxes, rpn_reg_pred, rpn_cls_pred, frcnn_reg_pred, frcnn_cls_pred = frcnn_pred
    frcnn_reg_pred = tf.reshape(frcnn_reg_pred, (batch_size, tf.shape(frcnn_reg_pred)[1], total_labels, 4))
    frcnn_reg_pred *= hyper_params["variances"]
    #
    expanded_roi_bboxes = tf.tile(tf.expand_dims(roi_bboxes, -2), (1, 1, total_labels, 1))
    frcnn_bboxes = bbox_utils.get_bboxes_from_deltas(expanded_roi_bboxes, frcnn_reg_pred)
    # We remove background predictions and reshape outputs for non max suppression
    frcnn_cls_pred = tf.cast(frcnn_cls_pred, tf.float32)
    pred_labels_map = tf.argmax(frcnn_cls_pred, 2, output_type=tf.int32)
    valid_cond = tf.not_equal(pred_labels_map, bg_id)
    #
    valid_bboxes = tf.expand_dims(frcnn_bboxes[valid_cond], 0)
    valid_labels = tf.expand_dims(frcnn_cls_pred[valid_cond], 0)
    #
    nms_bboxes, nmsed_scores, nmsed_classes, valid_detections = bbox_utils.non_max_suppression(valid_bboxes, valid_labels,
                                                                                               max_output_size_per_class=10,
                                                                                               max_total_size=200, score_threshold=0.6)
    denormalized_bboxes = bbox_utils.denormalize_bboxes(nms_bboxes[0], img_size, img_size)
    drawing_utils.draw_bboxes_with_labels(img[0], denormalized_bboxes, nmsed_classes[0], nmsed_scores[0], labels)
