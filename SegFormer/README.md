# SegFormer

SegFormer is a Transformer-based framework for semantic segmentation that unifies Transformers with lightweight multilayer perceptron (MLP) decoders. 

## Code Structure

```
- training.ipynb         # training model
- inference.ipynb        # inference for one test image
- inference_images       # containing one image and ground truth for inference
- inference_metric.ipynb # inference for evaluating metric
- segformer_model        # containing three trained models
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

We download the pretrained model of SegFormer from HuggingFace. The detail can be found [here](https://huggingface.co/docs/transformers/main/model_doc/segformer). We use the MiT-b0 Model.

Hyperparameters for training
```
batch_size = 8
lr=0.00006
epochs = 20
```


## Results

We get the IoU results on the test set:

|    Dataset     |   Forest  | Non-forest  |    mIoU   |
| :------------: | :-------: | :---------: | :-------: |
|      Gray      |   73.13   |    89.04    |    81.08  |
|      Gray      |   59.61   |    76.92    |    68.27  |
|      Gray      |   72.60   |    90.97    |    81.79  |




