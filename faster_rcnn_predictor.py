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
labels = data_utils.get_labels(dataset_info)
labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)
img_size = hyper_params["img_size"]

if use_custom_images:
    test_data = data_utils.get_image_data_from_folder(custom_image_path, img_size, img_size)
else:
    test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, evaluate=evaluate))
    padded_shapes, padding_values = data_utils.get_padded_batch_params()
    test_data = test_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
#
anchors = bbox_utils.generate_anchors(hyper_params)
rpn_model, feature_extractor = get_rpn_model(hyper_params)
frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params, mode="inference")
#
frcnn_model_path = io_utils.get_model_path("faster_rcnn", backbone)
frcnn_model.load_weights(frcnn_model_path)

stats = eval_utils.init_stats(labels)

for image_data in test_data:
    imgs, gt_boxes, gt_labels = image_data
    pred_bboxes, pred_labels, pred_scores = frcnn_model.predict_on_batch(imgs)
    if evaluate:
        stats = eval_utils.update_stats(pred_bboxes, pred_labels, pred_scores, gt_boxes, gt_labels, stats)
        continue
    for i, img in enumerate(imgs):
        denormalized_bboxes = bbox_utils.denormalize_bboxes(pred_bboxes[i], img_size, img_size)
        drawing_utils.draw_bboxes_with_labels(img, denormalized_bboxes, pred_labels[i], pred_scores[i], labels)

if evaluate:
    stats, mAP = eval_utils.calculate_mAP(stats)
    print("mAP: {}".format(float(mAP)))
