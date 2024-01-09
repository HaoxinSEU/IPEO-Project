# SAM model

- The SAM (Segment Anything Model) is a core component of our project. Install the SAM library directly from Facebook Research's repository: https://github.com/facebookresearch/segment-anything.git
- The Transformers library by Hugging Face provides a range of tools and pre-trained models. For our project, we are using a specific version that is compatible with our setup: https://github.com/huggingface/transformers.git@v4.36.1

## Description
- `interference.ipynb` imports the model with the trained weights, imports a test image and corresponding mask, then shows an image's segmentation result;
- `sam_model.ipynb` runs the model including data preprocessing, training process and interference.

## Folder structure
```
- SAM_training_model    # SAM model with training process and inference 
  - sam_train_model.ipynb
- test_images           # Images for inference demo
  - input_2780_2020.tif
  - target_2780_2020.tif
- inference.ipynb
- README.md
```

## Requirements
In order to run the code, please install the following packages:
```
numpy
matplotlib
tifffile
patchify
scipy
datasets (Hugging Face)
Pillow
torch (PyTorch)
transformers (Hugging Face)
albumentations
opencv-python (cv2)
scikit-learn

```

## Trained weights
We provide the trained weights when using _Gray_ dataset, as it gives the best IoU on forest. The weights can be found on [Google Drive](https://drive.google.com/drive/folders/1F9US5y1icSQR0qdV2zHNwm4Y99Lzuc_d?usp=drive_link)

## Results
When using the following hyperparameters:
```
batch_size = 5
epoch = 7
initial_learning_rate = 8e-4
```
We get the IoU results on the test set:

|    Dataset     |   Forest  | Non-forest  |    mIoU   |
| :------------: | :-------: | :---------: | :-------: |
|      Gray      | **52.93** |   75.46     |   64.20   |
|  Deoldify_part |   51.98   | **76.63**   | **64.31** |
|  Deoldify_all  |   45.13   |   75.28     |   60.20   |

