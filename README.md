# IPEO 2023 Project: Forest Classification for Swiss Alps

This is the repository for our IPEO 2023 Project, which does historical forest mapping, i.e., semantic segmentation on the Swiss Alps.

In this project, we implement four different models to do this task, namely U-Net, DeepLabv3, SegFormer and Segment Anything (SAM).


## Author
  - Cyril Golaz
  - Shengyu He
  - Haoxin Sun
  - Yujie Wu


## Project structure
This repository is organized as follows:
```
- DeepLabv3               # source code for DeepLabv3
  - sources
  - README.md
- Preprocessing           # source code for dataset processing
  - ConvertFormat.ipynb
  - DeOldify.ipynb
  - RemoveImages.ipynb
- SegFormer               # source code for SegFormer
  - sources
  - README.md             
- SAM                     # source code for SAM
  - sources               
  - README.md
- U-Net                   # source code for U-Net
  - sources
  - README.md
- README.md
```


## Dataset
The dataset contains both gray images and RGB images, and the ground truth contains two classes: non-forest (0) and forest (1). While some pixels are non-labeled, which are defined as 255 in the ground truth.

- We first remove images that more than 25% of their pixels are non-labeled in the training dataset (63 images)
- Then convert grayscale images to RGB images in three different ways:
    - Directly convert all grayscale images to RGB images by `PIL.Image.convert('RGB')`, i.e. images are still gray but contain 3 channels; No operation on original RGB images;
    - Use a DL-based method called [DeOldify](https://github.com/jantic/DeOldify) to do colorization on all grayscale images; No operation on original RGB images;
    - Use DeOldify to do colorization on all images, i.e., both grayscale images and RGB images.

You can find the processed datasets in these different ways [here](https://drive.google.com/drive/folders/1hEs_I2NBof5FsnYbACwwSZ5u08B1r-KB?usp=sharing).


## Data Preprocessing
### Deal with different image sizes
Since images are in different sizes, e.g., 256x256, 232x256, 256x232. So we use two different ways to make them the same size:
- Zero-padding to pad all images to the same size of 256x256, and the ground truth for the padding area is non-labeled;
- Directly resize all images and ground truth to 256x256, with interpolation mode NEAREST. 

### Data augmentation
When training for some models, we also have data augmentation by applying random vertical and horizon flips when loading images in the training process.

### Model-specific processing
For different models, we have some specific processing, the details of this part are illustrated in the reports.

### Class imbalance
To fight against class imbalance, we have the following methods for the training set and validation set respectively:
- For the training set, we resample images with more forest labels. After resampling, the ratio between #non-forest pixel and #forest pixels is 1.22;
- For the validation set, we remove the images with no forest labels.

Depending on the different models we implement, we determine whether to use these methods or not, to get the best performance.


## Training and Inference on Each Model
Please check the instructions in each model's folder, which contains the details to do training and inference on each implemented model respectively.


## Results
The IoU results of four models<sup>$\dagger$</sup> are shown below:

|     Model      |   Forest  | Non-forest  |    mIoU   |
| :------------: | :-------: | :---------: | :-------: |
|      U-Net     |    42.43  |    75.11    | **58.77** |
|    DeepLabv3   |    61.68  |    77.63    | **69.65** |
|    SegFormer   |    73.13  |    89.04    | **81.08** |
|       SAM      |    52.93  |    75.46    | **64.20** |

($\dagger$: the results are the best IoU we can get with different processed datasets for these models.)


## Acknowledgement
This project is based on some open-source code bases, such as U-Net, DeepLabv3, SegFormer, and SAM. Huge thanks to all the projects.
