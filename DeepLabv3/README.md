# DeepLabv3 for Historical Forest Mapping

This folder includes the source code to run forest mapping (semantic segmentation) by DeepLabv3. 


## Code structure
```
- demo_image            # demo image and ground truth for inference  
  - image.tif
  - target.tif
- sources               # source code
  - accuracy.py         # per-pixel accuracy calculation
  - custom_dataset.py   # dataset class
  - custom_model.py     # DeepLabv3 model
  - inference.ipynb     # demo notebook to run inference
  - iou.py              # IoU calculation
  - main_inference.py   # script to run inference
  - main_training.py    # script to run training
  - train.py            # training process
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
%run /PATH_TO_SOURCES/main_training.py /DATASET /WEIGHT_OUTPUT --keep_feature_extract --train_forest_balance --num_classes 3 --epochs 100 --batch_size 32 -w 1 -w 1.3 -w 0.4
```
There are also other arguments available when running the training, check the code for more details.

Alternatively, you can also run the script `main_training.py` from the terminal, with the same arguments.


## Trained weights
We provided the trained weights when using the dataset: __Forest_Deoldify_all__, which gives the best IoU for the forest class. The trained weights can be found [here](https://drive.google.com/file/d/1DfVE0w9eHPQx3NEKwiGiNQ3g4ZR8fhNI/view?usp=sharing). 


## Inference
To run the inference, in the notebook:
```
%run /PATH_TO_SOURCES/main_inference.py /DATASET /WEIGHTS/best_DeepLabV3_weight.pth --data_mode "test" --num_classes 3
```

And similar to the training, you can directly run the script `main_inference.py` in the terminal. 


## Results
When using the following hyperparameters:
```
batch_size = 32
epoch = 100
initial_learning_rate = 1e-5
weights = [1, 1.3, 0.4]
```
We get the IoU results on the test set:

|    Dataset     |   Forest  | Non-forest  |    mIoU   |
| :------------: | :-------: | :---------: | :-------: |
|      Gray      |   60.43   |  **79.49**  | **69.96** |
|  Deoldify_part |   59.95   |   77.94     |   68.95   |
|  Deoldify_all  | **61.68** |   77.63     |   69.65   |
