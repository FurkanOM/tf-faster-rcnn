import tensorflow as tf
from utils import io_utils, data_utils, train_utils, bbox_utils, drawing_utils, eval_utils
from models import faster_rcnn

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 4
evaluate = False
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
total_items = data_utils.get_total_item_size(dataset_info, "test")
labels = data_utils.get_labels(dataset_info)
labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)
img_size = hyper_params["img_size"]

data_types = data_utils.get_data_types()
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()

if use_custom_images:
    img_paths = data_utils.get_custom_imgs(custom_image_path)
    total_items = len(img_paths)
    test_data = tf.data.Dataset.from_generator(lambda: data_utils.custom_data_generator(
                                               img_paths, img_size, img_size), data_types, data_shapes)
else:
    test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, evaluate=evaluate))
#
test_data = test_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
#
anchors = bbox_utils.generate_anchors(hyper_params)
rpn_model, feature_extractor = get_rpn_model(hyper_params)
frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params, mode="inference")
#
frcnn_model_path = io_utils.get_model_path("faster_rcnn", backbone)
frcnn_model.load_weights(frcnn_model_path)

step_size = train_utils.get_step_size(total_items, batch_size)
pred_bboxes, pred_labels, pred_scores = frcnn_model.predict(test_data, steps=step_size, verbose=1)

if evaluate:
    eval_utils.evaluate_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)
else:
    drawing_utils.draw_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)
