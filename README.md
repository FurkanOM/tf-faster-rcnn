# Faster-RCNN

This is tensorflow Faster-RCNN implementation from scratch supporting to the batch processing.
All methods are tried to be created in the simplest way for easy understanding.
Most of the operations performed during the implementation were carried out as described in the [paper](https://arxiv.org/pdf/1506.01497.pdf) and [tf-rpn](https://github.com/FurkanOM/tf-rpn) repository.

It's implemented and tested with **tensorflow 2.0**

## Usage

Project models created in virtual environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
You can also create required virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

To create virtual environment (tensorflow-2 gpu environment):

```sh
conda env create -f environment.yml
```

To train Faster-RCNN model:

```sh
python faster_rcnn_trainer.py
```

If you have gpu issues you can use **-handle-gpu** flag:

```sh
python faster_rcnn_trainer.py -handle-gpu
```

You can also train RPN alone:

```sh
python rpn_trainer.py
```

You can also use **-handle-gpu** flag all of above processes.

## Todo

* [x] Batch support
* [ ] Predictors and test results
* [ ] Inline documentation
* [ ] Hyperparam management using command line
* [ ] Multiple backbone support ([ResNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet101), [MobileNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNet), etc.)

### References

* [Mask RCNN](https://github.com/matterport/Mask_RCNN) - especially for the construction of the model
* [keras-frcnn](https://github.com/small-yellow-duck/keras-frcnn)
* [PyTorch Faster RCNN](https://github.com/rbgirshick/py-faster-rcnn)
