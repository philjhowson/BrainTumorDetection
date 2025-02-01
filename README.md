# Machine Learning Models for Brain Tumor Detection
***
This project utilizes three different machine learning models to test their efficiency
at diagnosing the patients with brain tumors. The dataset used for training was retrieved
from Kaggle. Visit the link below for more information and to download the image files
used in model training:

[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data)

## Project Organization
------------------------------------------------------------------------
    root
    ├── data # contains indices for training, validation, and testing
    │   ├── test_indices.pkl
    │   ├── train_indices.pkl
    │   └── val_indices.pkl
    ├── images # contains output images from data exploration and model evaluation
    │   ├── custom_models_grad_cam_v1.png
    │   ├── custom_model_history_v1.png
    │   ├── custom_resnet50_grad_cam_v1.png
    │   ├── data_exploration.png
    │   ├── densenet_gradcam_v1.png
    │   ├── densenet_history_v1.png
    │   ├── model_comparison.png
    │   ├── resizing.png
    │   └── resnet_history_v1.png
    ├── metrics # output metrics for model training and evaluation
    │   ├── custom_model_classification_report.pkl
    │   ├── custom_model_scores.pkl
    │   ├── custom_model_tumor_performance_v1.pkl
    │   ├── densenet_classification_report.pkl
    │   ├── densenet_scores.pkl
    │   ├── densenet_tumor_performance_v1.pkl
    │   ├── resnet50_tumor_performance_v1.pkl
    │   ├── resnet_50_classification_report.pkl
    │   └── resnet_50_scores.pkl
    ├── models # output folder for model saves, this is not on github due to model sizes
    ├── scans # folder containing MRI scans, not on repository, download from Kaggle
    │   ├── no # contains images without a tumor 
    │   └── yes # contains images with a tumor 
    ├── src # contains source code for exploration, and model training and evaluation
    │   ├── compare_models.py
    │   ├── custom_functions.py
    │   ├── data_exploration.py
    │   ├── evaluate_custom_model.py 
    │   ├── evaluate_custom_resnet50.py 
    │   ├── evaluate_densenet.py 
    │   ├── split_data.py 
    │   ├── train_custom_model.py 
    │   ├── train_custom_resnet50.py 
    │   └── train_densenet.py 
    ├── streamlit # contains the files for the streamlit app
    │   ├── images # contains images used in the streamlit app
    │   ├── items # contains src files for the streamlit app 
    │   ├── metrics # contains metrics used in the streamlit app
    │   ├── app.py
    │   └── requirements.txt
    ├── .gitignore
    ├── main.py
    └── README.md

## Project Introduction
I began by exploring the dataset to determine the distribution of classes and the different
image properties. Then I visualized the distribution and characteristics of the dataset.
The training, validation, and test split were then created and saved for model training 
and evaluation.

Three models were build and tested using the [pytorch](https://pytorch.org/) library. I
utilized a ResNet50 model with additional layers and skip connections, DenseNet 162, and
a custom built model.

The ResNet50 model had three additional convolutional layers with skip connections that 
I trained on model specific tasks, while the majority of the ResNet50 model weights were
frozen to facilitate transfer learning. A small snippet of code below presents the 
general structure of the additional layers. There are a total of 91,838,657 parameters, 
of which 61,517,569 were trainable.

```python
    def forward(self, x):

        x = self.base_model(x)

        skip_connection = self.skip_connection1(x)
        x = self.Conv1(x)
        x = x + skip_connection
```

The DenseNet 162 model was unchanged, although I froze the majority of the layers, resulting
in 972,160 trainable parameters, out of a total 12,486,145.

The custom built model had a multi-branch architecture. Specifically, the input image was
sent through a block with a large kernel (kernel_size = 7) and a block with a medium kernel
(kernel = 5) size before being concatenated and fed through the remaining blocks.
The initial residual learning, was added to the second block and then residual
learning was continually accumulated in an intermediate object such that each block
had access to a feature map that contained the output of ever previous block. For
more details about the custom model architecture and the architecture of the additions
made to the ResNet50 model, please see the custom_functions.py file in the src folder.

```python
    def forward(self, x):

        large = self.large(x)
        medium = self.medium(x)
        x = torch.cat([large, medium], axis = 1)
        
        x = self.block1(x)

        residual = self.residual1(x)
        x = self.block2(x)
        x = x + residual

        intermediate = self.intermediate1(residual)
        residual = self.residual2(x)
        intermediate = torch.cat([intermediate, residual], axis = 1)
        x = self.block3(x)
        x = x + intermediate
```