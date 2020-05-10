import tensorflow as tf
from utils import io_utils, data_utils, train_utils, bbox_utils, drawing_utils

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 1
# If you have trained faster rcnn model you can load weights from faster rcnn model
load_weights_from_frcnn = False
backbone = args.backbone
io_utils.is_valid_backbone(backbone)

if backbone == "vgg16":
    from models.rpn_vgg16 import get_model

hyper_params = train_utils.get_hyper_params(backbone)

test_data, dataset_info = data_utils.get_dataset("voc/2007", "test")
labels = data_utils.get_labels(dataset_info)
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1
#
img_size = hyper_params["img_size"]
test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))

padded_shapes, padding_values = data_utils.get_padded_batch_params()
test_data = test_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

rpn_model, _ = get_model(hyper_params)

frcnn_model_path = io_utils.get_model_path("faster_rcnn")
rpn_model_path = io_utils.get_model_path("rpn")
model_path = frcnn_model_path if load_weights_from_frcnn else rpn_model_path
rpn_model.load_weights(model_path, by_name=True)

anchors = bbox_utils.generate_anchors(hyper_params)

for image_data in test_data:
    img, _, _ = image_data
    rpn_bbox_deltas, rpn_labels = rpn_model.predict_on_batch(img)
    #
    rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, -1, 4))
    rpn_bbox_deltas *= hyper_params["variances"]
    rpn_labels = tf.reshape(rpn_labels, (batch_size, -1, 1))
    #
    rpn_bboxes = bbox_utils.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)
    rpn_bboxes = tf.reshape(rpn_bboxes, (batch_size, -1, 1, 4))
    #
    nms_bboxes, _, _, _ = bbox_utils.non_max_suppression(rpn_bboxes, rpn_labels,
                                                         max_output_size_per_class=hyper_params["test_nms_topn"],
                                                         max_total_size=hyper_params["test_nms_topn"])
    drawing_utils.draw_bboxes(img, nms_bboxes)
