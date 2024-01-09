# UNet for Historical Forest Mapping

This folder includes the source code to run forest mapping (semantic segmentation) by UNet. 


## Code structure
```
- demo_image            # demo image and ground truth for inference  
  - image.tif
  - target.tif
- source               # source code
  - accuracy.py         # per-pixel accuracy calculation
  - inference.ipynb     # demo notebook to run inference
  - iou.py              # IoU calculation
  - main_inference.py   # script to run inference
  - train.py            # training process
  - dataloader.py       # to load the images
  - unet
    - unet_model.py       # UNet model
    - unet_parts.py       # the different parts of the UNet model
  - utils
    - dice_score.py       # to compute the dice score during the training
- README.md
```


## Requirements
In order to run the code, please install the following packages:
```
torch 
torchvision 
tqdm 
numpy 
pillow
```

## Training
To run the training, in the notebook:
```
%run /PATH_TO_SOURCE/train.py --classes 3 --batch-size 4 --epochs 10
```
There are also other arguments available when running the training, check the code for more details.

Alternatively, you can also run the script `train.py` from the terminal, with the same arguments.


## Trained weights
We provided the trained weights when using the dataset: __Forest_Deoldify_all__, which gives the best IoU for the forest class. The trained weights can be found [here](https://drive.google.com/drive/folders/1lzaWNAbFJFOS_UZ81nRmScvMbTrw6vp0?usp=drive_link). 


## Inference
To run the inference, in the notebook:
```
%run /PATH_TO_SOURCES/main_inference.py /DATASET /WEIGHTS/model_UNet.pth --data_mode "test" --num_classes 3
```

And similar to the training, you can directly run the script `main_inference.py` in the terminal. 


## Results
When using the following hyperparameters:
```
batch_size = 4
epoch = 10
initial_learning_rate = 1e-5
```
We get the IoU results on the test set:

|    Dataset     |   Forest  | Non-forest  |    mIoU   |
| :------------: | :-------: | :---------: | :-------: |
|      Gray      |   42.43   |   75.11     |   58.77   |     |
|  Deoldify_part |   40.00   |   74.24     |   57.1    |
|  Deoldify_all  | **44.17** | **75.17**   |  **59.67**|
