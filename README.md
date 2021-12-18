# CoraNet
This is the 3D Implementation of 《Inconsistency-aware Uncertainty Estimation for Semi-supervised Medical Image Segmentation》

## Environment

pytorch 3.7; pytorch 1.1.0; torchvision 0.4.2

## DataSets

You can download the dataset from [CT-Pancreas](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT).
The data structure is as follows:
```
pancreas
├── Pancreas-CT
└── Pancreas-CT-Label
```

And do the preprocessing with ``preprocess/pancreas_preprocess.py`` for CT-Pancreas dataset.

## Pretrained Model

The pretrained VNet model on Pancreas-CT dataset can be dowload [here](https://drive.google.com/file/d/1rg0NmWl-UXa4aMNIK6gW58xg5P610jL0/view?usp=sharing)
