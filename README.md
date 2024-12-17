# Machine Learning Models for Brain Tumor Diagnosis
The objective of this project is to utilize cutting edge machine 
learning technology to train models to detect brain tumors based 
on MRI images. In this project, I make use of three different models,
ResNet50, DenseNet, and a custom made model to assess their ability
to detect brain tumors in MRI images.

## The Dataset
Dataset retrieved from kaggle:

[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data)

This dataset contrains MRI images of brains with and without tumors.
The objective of this project is to build a neural network that can
detect the presence of a brain tumor based on an MRI scans. To achieve
this objective, I trained three different models to evaluate
performance.

## The Models

1. ResNet50 ([He et al., 2015](https://arxiv.org/abs/1512.03385))

Resnet50 utilizes residual learning through "skip connections" to
faciliate deeper neural networks without loss of accuracy which
occured in earlier models.

2. DenseNet ([Huang et al., 2018](https://arxiv.org/abs/1608.06993))

DenseNet differs from other models in that feature maps from all
previous layers of the model are shared with each layer. This
interconnectness facilitates deeper neural networks with less
computational load than other models.

3. Custom Model

I built a custom model that utilizes different kernel sizes to
leverage their ability to capture local and global relationships
before concatinating the feature maps for pass through deeper
convolutional layers. Each block in the network shares the output
feature map of the previous blocks to facilitate a deeper
neural network much like ResNet50 and DenseNet.