# PyTorch-DeepDream
A simplified implementation of the DeepDream algorithm using PyTorch. Based on https://hackernoon.com/dl06-deepdream-with-code-5f735052e21f

## Dependencies

* torch
* torchvision
* numpy
* matplotlib
* PIL (Pillow)

## Usage

deep_dream.py [-f FILENAME]

### Optional Arguments:

-h, --help  show this help message and exit

-l LAYER    Desired network layer to push image through to. (Default = 16)

-i ITER     Desired iterations to apply to image. (Default = 11)

-lr LR      Desired learning rate to use in algorithm. (Default = 0.0012)
