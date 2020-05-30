# Faster-RCNN

This is tensorflow Faster-RCNN implementation from scratch supporting to the batch processing.
All methods are tried to be created in the simplest way for easy understanding.
Most of the operations performed during the implementation were carried out as described in the [paper](https://arxiv.org/abs/1506.01497) and [tf-rpn](https://github.com/FurkanOM/tf-rpn) repository.

It's implemented and tested with **tensorflow 2.0**.

[MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) and [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16) backbones are supported.

## Usage

Project models created in virtual environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
You can also create required virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

To create virtual environment (tensorflow-2 gpu environment):

```sh
conda env create -f environment.yml
```

There are two different backbone, first one the legacy **vgg16** backbone and the second and default one is **mobilenet_v2**.
You can easily specify the backbone to be used with the **--backbone** parameter.
Default backbone is **mobilenet_v2**.

To train and test Faster-RCNN model:

```sh
python faster_rcnn_trainer.py --backbone mobilenet_v2
python faster_rcnn_predictor.py --backbone mobilenet_v2
```

You can also train and test RPN alone:

```sh
python rpn_trainer.py --backbone vgg16
python rpn_predictor.py --backbone vgg16
```

If you have GPU issues you can use **-handle-gpu** flag with these commands:

```sh
python faster_rcnn_trainer.py -handle-gpu
```

## Examples

| Trained with VOC 0712 trainval data |
| -------------- |
| ![Man riding bike](http://furkanomerustaoglu.com/wp-content/uploads/2020/05/man_with_bike_faster_rcnn.png) |
| Photo by William Peynichou on Unsplash |
| ![Airplanes](http://furkanomerustaoglu.com/wp-content/uploads/2020/05/planes_faster_rcnn.png) |
| Photo by Vishu Gowda on Unsplash |

### References

* VOC 2007 Dataset [[dataset]](http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html)
* VOC 2012 Dataset [[dataset]](http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html)
* Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks [[paper]](https://arxiv.org/abs/1506.01497)
* Object Detection Metrics [[code]](https://github.com/rafaelpadilla/Object-Detection-Metrics)
* Mask RCNN [[code]](https://github.com/matterport/Mask_RCNN)
* keras-frcnn [[code]](https://github.com/small-yellow-duck/keras-frcnn)
* PyTorch Faster RCNN [[code]](https://github.com/rbgirshick/py-faster-rcnn)
