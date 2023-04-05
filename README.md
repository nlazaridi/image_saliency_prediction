# Saliency Prediction in Images

This repository contains code for saliency prediction in images using [VST model](https://arxiv.org/abs/2104.12099). The code is written in Python and uses PyTorch for training and inference.

## Downloading the Pretrained Model

A pretrained model is available for download from the following link:

[drive](https://drive.google.com/file/d/1RWyrU72GPgAFdcglI1V_DLHWPuiRiTHM/view?usp=sharing)

To use this model, simply download the .pt file and load it into your PyTorch code.

## Installing the Conda Environment

To use this code, you need to install the required dependencies using Conda. Here's how to create a new Conda environment and install the dependencies:

1. Clone this repository to your local machine.
2. Open a terminal or command prompt and navigate to the root directory of the cloned repository.
3. Create a new Conda environment using the following command:
```console
conda env create -f sal_pred.yml
```
4. Activate the new Conda environment using the following command:
```console
conda activate sal_pred_env
```

## Usage

To use this code, first activate the Conda environment as described above. Then, you can run the code using the following command:
```console
python generate_mask.py --img_path='/images/0245.png' --save_test_path_root='preds/'
```
Replace '/images/0245.png' with the path to the input image that you want to predict saliency for and 'preds/' with the path to the folder that saliency mask will be saved.

