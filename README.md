# Faster-RCNN

This is tensorflow Faster-RCNN implementation from scratch supporting to the batch processing.
All methods are tried to be created in the simplest way for easy understanding.
Most of the operations performed during the implementation were carried out as described in the [paper](https://arxiv.org/abs/1506.01497) and [tf-rpn](https://github.com/FurkanOM/tf-rpn) repository.

It's implemented and tested with **tensorflow 2.0**

## Usage

Project models created in virtual environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
You can also create required virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

To create virtual environment (tensorflow-2 gpu environment):

```sh
conda env create -f environment.yml
```

To train and test Faster-RCNN model:

```sh
python faster_rcnn_trainer.py
python faster_rcnn_predictor.py
```

You can also train and test RPN alone:

```sh
python rpn_trainer.py
python rpn_predictor.py
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

## Todo

* [x] Batch support
* [x] Predictors and test results
* [x] Inline documentation
* [ ] MobileNetV2 backbone support ([MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2))

### References

* [VOC 2007 Dataset](http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html)
* [Mask RCNN](https://github.com/matterport/Mask_RCNN) - especially for the construction of the model
* [keras-frcnn](https://github.com/small-yellow-duck/keras-frcnn)
* [PyTorch Faster RCNN](https://github.com/rbgirshick/py-faster-rcnn)
