# Machine Learning Models for Brain Tumor Detection
***
This project utilizes three different machine learning models to test their efficiency
at diagnosing the patients with brain tumors. The dataset used for training was retrieved
from kaggle. Visit the link below for more information and to download the image files
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
    │   ├── custom_model_grad_cam_v2.png
    │   ├── custom_model_history_v2.png
    │   ├── data_exploration.png
    │   ├── densenet_gradcam_v2.png
    │   ├── densenet_history_v2.png
    │   ├── model_comparison_2_2_2.png
    │   ├── resizing.png
    │   ├── resnet50_grad_cam_v2.png
    │   └── resnet50_history_v2.png
    ├── metrics # output metrics for model training and evaluation
    │   ├── custom_model_classification_report_v2.pkl
    │   ├── custom_model_scores_v2.pkl
    │   ├── custom_model_tumor_performance_v2.pkl
    │   ├── densenet_classification_report_v2.pkl
    │   ├── densenet_scores_v2.pkl
    │   ├── densenet_tumor_performance_v2.pkl
    │   ├── resnet50_classification_report_v2.pkl
    │   ├── resnet50_scores_v2.pkl
    │   └── resnet50_tumor_performance_v2.pkl
    ├── models # output folder for model saves, this is not on github due to model sizes
    ├── scans # folder containing MRI scans, not on repository, download from kaggle
    │   ├── no # contains images without a tumor 
    │   └── yes # contains images with a tumor 
    ├── src # contains source code for exploration, and model training and evaluation
    │   ├── compare_models.py
    │   ├── custom_functions.py
    │   ├── data_exploration.py
    │   ├── evaluate_custom_model.py
    │   ├── evaluate_densenet.py  
    │   ├── evaluate_resnet50.py 
    │   ├── split_data.py 
    │   ├── train_custom_model.py 
    │   ├── train_densenet.py 
    │   └── train_resnet50.py 
    ├── streamlit # contains the files for the streamlit app
    │   ├── images # contains images used in the streamlit app
    │   ├── items # contains src files for the streamlit app 
    │   ├── metrics # contains metrics used in the streamlit app
    │   ├── app.py
    │   └── requirements.txt
    ├── .gitignore
    ├── LICENSE
    ├── main.py
    ├── README.md
    └── requirements.txt

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

The DenseNet162 model was unchanged, although I froze the majority of the layers, resulting
in 972,160 trainable parameters, out of a total 12,486,145.

The custom built model had a multi-branch architecture. Specifically, the input image was
sent through a block with a large kernel (kernel_size = 7) and a block with a medium kernel
(kernel_size = 5) size before being concatenated and fed through the remaining blocks.
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

## Getting Started

Clone the repository:

```sh
git clone https://github.com/philjhowson/BrainTumorDetection
```

Navigate to the cloned repository and create a folders for the models and scans:

```sh
mkdir models scans
```
Download the scans from [kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data)
 and unpack the yes and no folders from the zip into the scans folder.

 The scripts assume the root directory of the repository is the directory you are running
 the code from.

 Everything can be run from the main.py file, which has one mandatory argument, when
 being run --script, which should be one of explore, preparation, train, or compare.
 However, for all options besides explore and prepare, at least one model needs
 to be specified, using the --model argument. This argument takes either resnet, densenet,
 or custom as arguments. You can also optionally include a version argument --version,
 which expects and int, such as 1, or 2. This can be useful if you want to run the
 model training more than once but to save all the outputs individual. The default
 version is 1.

 ```sh
 python main.py --script explore
 ```
```sh
python main.py --script train --model densenet --version 2
```

The model comparison script also has two arguments, --models and --versions, which take
a minimum of one argument, but can take more. The default arguments for --models
resnet, densenet, custom and the default for --versions is 1, 1, 1,.

```sh
python main.py --script compare --models resnet densenet custom --versions 3 1 2
```

The training script will run both the training and evaluation scripts, but each
script can be run alone, but the script still expects it to be run from the root
directory. In these cases, the training and evaluation scripts only take the --version
argument, and default to 1 if not specified.

```sh
python src/evaluate_densenet.py --version 2
```

However, the model comparison script still takes the --models and --versions arguments
as with the main.py script.

```sh
python src/compare_models.py --models resnet custom --versions 2 1
```

## Future Directions

Machine learning models have a tremendous potential to assist diagnosis of a wide
range of illnesses. I plan to utilize more medical images from a wider range of
techniques in order to train further models to detect cancer and other illnesses.
I am also planning to use other openly available medical information to train
models on datasets such as blood work and other numerical based evaluations of
health.