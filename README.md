# Industrial Surface Defect Classification using CNN and ResNet18

A deep learning project for classifying steel surface defects into six categories using a custom convolutional neural network (CNN) and transfer learning with an ImageNet-pretrained ResNet18 model.

## Overview

Automated steel surface inspection can help identify manufacturing defects. This project compares a CNN trained from scratch with a pretrained ResNet18 feature extractor on the NEU surface defect image dataset.

**Task:** Image classification (not object detection or segmentation).

## Dataset

The NEU surface defect images cover six classes:

- Crazing
- Inclusion
- Patches
- Pitted surface
- Rolled-in scale
- Scratches

| Split | Images |
| --- | ---: |
| Training | 1,152 |
| Validation | 288 |
| Test | 360 |
| **Total** | **1,800** |

The training and validation subsets were stratified from the original training directory. The original validation directory was used as the test set.

## Methodology

1. **Preprocessing:** Convert grayscale images to three channels, resize to 224 × 224, and normalize for the pretrained ResNet18 model.
2. **Baseline:** Train a custom CNN with three convolutional blocks.
3. **Transfer learning:** Load ImageNet-pretrained ResNet18, freeze its pretrained feature-extraction layers, and replace/train the final fully connected layer for six classes.
4. **Evaluation:** Compare models on the held-out test split using accuracy and macro F1-score; examine per-class precision, recall, and confusion matrices.

## Results

| Model | Test accuracy | Macro F1-score |
| --- | ---: | ---: |
| Custom CNN | 82.78% | 82.42% |
| Pretrained ResNet18 | **98.61%** | **98.61%** |

ResNet18 correctly classified **355 of 360** test images in this experiment.

## Repository contents

- `notebooks/steel_defect_classification.ipynb` — data preparation, training, and evaluation (add your Colab notebook here)
- `requirements.txt` — Python dependencies
- `models/resnet18_steel_defects.pth` — optional saved checkpoint (add separately if distributing weights)

## Getting started

1. Obtain the NEU steel surface defect image dataset from its authorized distribution source.
2. Clone this repository and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `notebooks/steel_defect_classification.ipynb` in Jupyter or Google Colab.
4. Update the dataset directory paths in the notebook to match your environment, then run the cells in order.

> The notebook may contain Colab-specific paths. Dataset files and trained model weights are not bundled with this README.

## Limitations

- A SHA-256 check identified **one byte-identical image pair across the training and validation splits** (`patches_105.jpg` and `patches_101.jpg`).
- No byte-identical overlap was found between the test split and the training or validation splits. Near-duplicate analysis was not performed.
- These results describe performance on the NEU dataset; they do not establish accuracy on new industrial production data.

## Possible future work

- Rebuild leakage-free dataset splits and repeat evaluation.
- Test generalization on images from other steel inspection environments.
- Compare partial fine-tuning with frozen-feature transfer learning.
- Add a simple inference interface for new images.

## Tech stack

Python · PyTorch · torchvision · scikit-learn · NumPy · Matplotlib · Google Colab
