"""Run Faster R-CNN inference or evaluation on Pascal VOC data."""

from __future__ import annotations

from utils import io_utils, data_utils, train_utils, bbox_utils, drawing_utils, eval_utils
from models import faster_rcnn


def main() -> None:
    """Run Faster R-CNN inference or evaluation from the command line.

    Returns:
        None: Predictions are evaluated or rendered to the screen.
    """
    args = io_utils.handle_args()
    if args.handle_gpu:
        io_utils.handle_gpu_compatibility()

    batch_size = 4
    evaluate = False
    use_custom_images = False
    custom_image_path = "data/images/"
    backbone = args.backbone
    get_rpn_model = io_utils.get_rpn_model_builder(backbone)

    hyper_params = train_utils.get_hyper_params(backbone)

    test_data, dataset_info = data_utils.get_dataset("voc/2007", "test")
    total_items = data_utils.get_total_item_size(dataset_info, "test")
    labels = data_utils.get_labels(dataset_info)
    labels = ["bg"] + labels
    hyper_params["total_labels"] = len(labels)
    img_size = hyper_params["img_size"]

    if use_custom_images:
        img_paths = data_utils.get_custom_imgs(custom_image_path)
        total_items = len(img_paths)
        test_data = data_utils.build_custom_dataset(img_paths, img_size, img_size)
    else:
        test_data = data_utils.build_dataset(test_data, img_size, img_size, batch_size, evaluate=evaluate)

    if use_custom_images:
        test_data = test_data.padded_batch(
            batch_size,
            padded_shapes=data_utils.get_data_shapes(),
            padding_values=data_utils.get_padding_values()
        )

    anchors = bbox_utils.generate_anchors(hyper_params)
    rpn_model, feature_extractor = get_rpn_model(hyper_params)
    frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params, mode="inference")
    frcnn_model_path = io_utils.get_model_path("faster_rcnn", backbone)
    frcnn_model.load_weights(frcnn_model_path)

    step_size = train_utils.get_step_size(total_items, batch_size)
    pred_bboxes, pred_labels, pred_scores = frcnn_model.predict(test_data, steps=step_size, verbose=1)

    if evaluate:
        eval_utils.evaluate_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)
    else:
        drawing_utils.draw_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)


if __name__ == "__main__":
    main()
