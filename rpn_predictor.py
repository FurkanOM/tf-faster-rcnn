import tensorflow as tf
from utils import io_utils, data_utils, train_utils, bbox_utils, drawing_utils

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 4
use_custom_images = False
custom_image_path = "data/images/"
# If you have trained faster rcnn model you can load weights from faster rcnn model
load_weights_from_frcnn = False
backbone = args.backbone
io_utils.is_valid_backbone(backbone)

if backbone == "mobilenet_v2":
    from models.rpn_mobilenet_v2 import get_model
else:
    from models.rpn_vgg16 import get_model

hyper_params = train_utils.get_hyper_params(backbone)

test_data, dataset_info = data_utils.get_dataset("voc/2007", "test")
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
    test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))
#
test_data = test_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

rpn_model, _ = get_model(hyper_params)

frcnn_model_path = io_utils.get_model_path("faster_rcnn", backbone)
rpn_model_path = io_utils.get_model_path("rpn", backbone)
model_path = frcnn_model_path if load_weights_from_frcnn else rpn_model_path
rpn_model.load_weights(model_path, by_name=True)

anchors = bbox_utils.generate_anchors(hyper_params)

for image_data in test_data:
    imgs, _, _ = image_data
    rpn_bbox_deltas, rpn_labels = rpn_model.predict_on_batch(imgs)
    #
    rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, -1, 4))
    rpn_labels = tf.reshape(rpn_labels, (batch_size, -1))
    #
    rpn_bbox_deltas *= hyper_params["variances"]
    rpn_bboxes = bbox_utils.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)
    #
    _, top_indices = tf.nn.top_k(rpn_labels, 10)
    #
    selected_rpn_bboxes = tf.gather(rpn_bboxes, top_indices, batch_dims=1)
    #
    drawing_utils.draw_bboxes(imgs, selected_rpn_bboxes)
