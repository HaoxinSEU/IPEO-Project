# SegFormer

SegFormer is a Transformer-based framework for semantic segmentation that unifies Transformers with lightweight multilayer perceptron (MLP) decoders. 

## Code Structure

```
- segformer_training.ipynb   # for training model
- segformer_inference.ipynb  # for inference 
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

10 epochs, 10 mins per each on average in Kaggle Notebook


## Results

We get the IoU results on the test set:

|    Dataset     |   Forest  | Non-forest  |    mIoU   |
| :------------: | :-------: | :---------: | :-------: |
|      Gray      |   68.96   |    80.34    |    80.46  |



