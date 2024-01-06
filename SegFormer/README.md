# SegFormer

SegFormer is a Transformer-based framework for semantic segmentation that unifies Transformers with lightweight multilayer perceptron (MLP) decoders. 

## Code Structure

```
- training.ipynb   # training model
- inference.ipynb  # inference
- segformer_model  # containing three trained models
```

## Requirements
Following moodle are required to run the codes

```
numpy
torch
matplotlib
albumentations
transformers
sklearn
datasets
```

## Training

learning rate 0.00006, 20 epochs, around 8 mins per each in Kaggle Notebook


## Results

We get the IoU results on the test set:

|    Dataset     |   Forest  | Non-forest  |    mIoU   |
| :------------: | :-------: | :---------: | :-------: |
|      Gray      |   68.96   |    80.34    |    80.46  |



