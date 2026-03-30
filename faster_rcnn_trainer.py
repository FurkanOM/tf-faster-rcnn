"""Train a Faster R-CNN model on Pascal VOC data."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from utils import io_utils, data_utils, train_utils, bbox_utils
from models import faster_rcnn


def main() -> None:
    """Run Faster R-CNN training from the command line.

    Returns:
        None: The model is trained and checkpoints are written to disk.
    """
    args = io_utils.handle_args()
    if args.handle_gpu:
        io_utils.handle_gpu_compatibility()

    batch_size = 4
    epochs = 50
    load_weights = False
    with_voc_2012 = True
    backbone = args.backbone
    get_rpn_model = io_utils.get_rpn_model_builder(backbone)

    hyper_params = train_utils.get_hyper_params(backbone)

    train_data, dataset_info = data_utils.get_dataset("voc/2007", "train+validation")
    val_data, _ = data_utils.get_dataset("voc/2007", "test")
    train_total_items = data_utils.get_total_item_size(dataset_info, "train+validation")
    val_total_items = data_utils.get_total_item_size(dataset_info, "test")

    if with_voc_2012:
        voc_2012_data, voc_2012_info = data_utils.get_dataset("voc/2012", "train+validation")
        voc_2012_total_items = data_utils.get_total_item_size(voc_2012_info, "train+validation")
        train_total_items += voc_2012_total_items
        train_data = train_data.concatenate(voc_2012_data)

    labels = data_utils.get_labels(dataset_info)
    hyper_params["total_labels"] = len(labels) + 1
    img_size = hyper_params["img_size"]
    train_data = data_utils.build_dataset(
        train_data,
        img_size,
        img_size,
        batch_size,
        apply_augmentation=True
    )
    val_data = data_utils.build_dataset(val_data, img_size, img_size, batch_size)

    anchors = bbox_utils.generate_anchors(hyper_params)
    frcnn_train_feed = train_utils.faster_rcnn_generator(train_data, anchors, hyper_params)
    frcnn_val_feed = train_utils.faster_rcnn_generator(val_data, anchors, hyper_params)
    rpn_model, feature_extractor = get_rpn_model(hyper_params)
    frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params)
    frcnn_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-5),
        # Losses are attached inside the model graph, so Keras still expects one
        # placeholder entry per output even though the real supervision lives in
        # `add_loss(...)`.
        loss=[None] * len(frcnn_model.output)
    )
    faster_rcnn.init_model(frcnn_model, hyper_params)

    rpn_load_weights = False
    if rpn_load_weights:
        rpn_model_path = io_utils.get_model_path("rpn", backbone)
        rpn_model.load_weights(rpn_model_path)

    frcnn_model_path = io_utils.get_model_path("faster_rcnn", backbone)

    if load_weights:
        frcnn_model.load_weights(frcnn_model_path)
    log_path = io_utils.get_log_path("faster_rcnn", backbone)

    checkpoint_callback = ModelCheckpoint(
        frcnn_model_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True
    )
    tensorboard_callback = TensorBoard(log_dir=log_path)

    step_size_train = train_utils.get_step_size(train_total_items, batch_size)
    step_size_val = train_utils.get_step_size(val_total_items, batch_size)
    frcnn_model.fit(
        frcnn_train_feed,
        steps_per_epoch=step_size_train,
        validation_data=frcnn_val_feed,
        validation_steps=step_size_val,
        epochs=epochs,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )


if __name__ == "__main__":
    main()
