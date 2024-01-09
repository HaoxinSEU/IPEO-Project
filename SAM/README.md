# SAM model

- The SAM (Segment Anything Model) is a core component of our project. Install the SAM library directly from Facebook Research's repository: https://github.com/facebookresearch/segment-anything.git
- The Transformers library by Hugging Face provides a range of tools and pre-trained models. For our project, we are using a specific version that is compatible with our setup: https://github.com/huggingface/transformers.git@v4.36.1

## Description
- `interference.ipynb` import the model `SAM_model_7_epoch_0.5_threshold` with epoch 7, threshold 0.5 on Google drive (https://drive.google.com/file/d/13vOUt-APUB0M54EFmrcmYx3LnVSHn5cr/view?usp=sharing), import a test image and corresponding mask, then shows an image's segmentation result;
- `sam_model.ipynb` runs the model including data preprocessing, training process and interference;
