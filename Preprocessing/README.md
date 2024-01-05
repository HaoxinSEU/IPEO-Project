# Dataset processing

This folder contains three python notebook files, which process the dataset as described in README.

## Description
- `ConvertFormat.ipynb` converts the original gray images (only one channel) to 3-channel RGB images, without changing the color (still gray);
- `Deoldify.ipynb` runs the Deolify model, and colorizes the input images;
- `RemoveImages.ipynb` removes the images that more than 30% non-labeled pixels in the training dataset.

## Acknowledgements
For `Deoldify.ipynb`, it's based on the provided notebook from [Deoldify](https://github.com/jantic/DeOldify), great thanks to this open-source project!