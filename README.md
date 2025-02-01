# Machine Learning Models for Brain Tumor Detection
***
This project demonstrates MLOps best practices using a machine learning 
model that predicts if an NBA player will make a specific shot or not. The emphasis
is less on the machine learning model, and more on the automated data pipeline to train
the model, the deployment of the model, and the monitoring of the model.

## Project Organization
------------------------------------------------------------------------
    Root
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
    │   ├── resnet_50_scores.pkl
    ├── models # output folder for model saves, this is not on github due to model sizes
    ├── scans # folder containing MRI scans, not on repository, download from Kaggle
    │   ├── no
    │   ├── yes
    ├── src # contains source code for exploration, and model training and evaluation
    │   ├── compare_models.py
    │   ├── custom_functions.py
    │   ├── data_exploration.py
    │   ├── evaluate_custom_model.py 
    │   ├── evaluate_custom_resnet50.py 
    │   ├── evaluate_densenet.py 
    │   ├──
    │   ├──
    │   ├──
    │   ├──
    ├── 