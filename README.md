# IPEO 2023 Project: Forest Classification for Swiss Alps

This is the repo for our IPEO 2023 Project. 

## Author
  - Cyril Golaz
  - Shengyu He
  - Haoxin Sun
  - Yujie Wu

## Dataset
We first remove images that have more than 30% non-labeled pixels in the training dataset (69 images), then convert grayscale images to RGB images in three different ways.
- Directly convert all grayscale images to RGB images by `PIL.Image.convert('RGB')`, i.e. images are still gray but contain 3 channels; No operation on original RGB images
- Use a DL-based method called [DeOldify](https://github.com/jantic/DeOldify) to do colorization on all grayscale images; No operation on original RGB images
- Use DeOldify to do colorization on all images, i.e., both grayscale images and RGB images

You can find the processed datasets in these different ways [here](https://drive.google.com/drive/folders/1hEs_I2NBof5FsnYbACwwSZ5u08B1r-KB?usp=sharing).

## How to use Dataset class
- Copy `./DeepLabV3FineTuning/sources/dataloader.py` into your folder
- Initialize a `DataLoaderSegmentation` object with the _path_ to the dataset, and the _mode_
- Initialize a `torch.utils.data.DataLoader` with the object in the previous step
- Now you can use this `DataLoader` to generate an iterator (containing images and labels) for training, validation, and testing.
  
An example is in `./DeepLabV3FineTuning/sources/main_training.py:29~30` and  `./DeepLabV3FineTuning/sources/train.py:36`.


## Question
When doing the test, should we use the original size of images and ground truth?
Because now to run the pre-trained model, we have to resize images to 512*512. When measuring the accuracy and IoU, should we resize the output to 256*256 or 224*256, and compare with the original ground truth?
_potential problem_: GT size is different for different images, DataLoader can't use batch size > 1. Otherwise how to stack two GT together... Then it's very slow to do the test
