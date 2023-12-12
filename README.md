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

## Data Preprocessing
Since images are in different sizes, e.g., 256x256, 232x256, 256x232. So we use zero-padding to pad all images to the same size of 256x256. 

We also have data augmentation by applying random vertical and horizon flips when loading images.

## Class Imbalance
To fight against class imbalance, we resample images with more forest labels. After resampling, the ratio between #non-forest pixel and #forest pixels is 1.14

## How to use Dataset class
- Copy `./DeepLabV3FineTuning/sources/dataloader.py` into your folder
- Initialize a `DataLoaderSegmentation` object with the _path_ to the dataset, and the _mode_
- Initialize a `torch.utils.data.DataLoader` with the object in the previous step
- Now you can use this `DataLoader` to generate an iterator (containing images and labels) for training, validation, and testing.
  
An example is in `./DeepLabV3FineTuning/sources/main_training.py:29~30` and  `./DeepLabV3FineTuning/sources/train.py:36`.


