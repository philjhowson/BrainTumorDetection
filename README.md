C:.
|   .gitignore
|   README.md
|   tree.txt
|   
+---data
|       test_indices.pkl
|       train_indices.pkl
|       val_indices.pkl
|       
+---images
|       custom_models_grad_cam_v1.png
|       custom_model_history_v1.png
|       custom_resnet50_grad_cam_v1.png
|       data_exploration.png
|       densenet_gradcam_v1.png
|       densenet_history_v1.png
|       model_comparison.png
|       resizing.png
|       resnet_history_v1.png
|       
+---metrics
|       custom_model_classification_report.pkl
|       custom_model_scores.pkl
|       custom_model_tumor_performance_v1.pkl
|       densenet_classification_report.pkl
|       densenet_scores.pkl
|       densenet_tumor_performance_v1.pkl
|       resnet50_tumor_performance_v1.pkl
|       resnet_50_classification_report.pkl
|       resnet_50_scores.pkl
|       
+---models
|       custom_model_tumor_checkpoint_v1.pth
|       densenet_tumor_checkpoint_20.pth
|       densenet_tumor_checkpoint_v1.pth
|       resnet50_tumor_checkpoint_v1.pth
|       resnet50_tumor_checkpoint_v2.pth
|       resnet50_tumor_checkpoint_v3.pth
|       
+---scans
|   +---no
|   |       
|   \---yes
|           
+---src
|   |   compare_models.py
|   |   custom_functions.py
|   |   data_exploration.py
|   |   evaluate_custom_model.py
|   |   evaluate_custom_resnet50.py
|   |   evaluate_densenet.py
|   |   split_data.py
|   |   train_custom_model.py
|   |   train_custom_resnet50.py
|   |   train_densenet.py
|   |   
|   +---.ipynb_checkpoints
|   |       data_exploration-checkpoint.ipynb
|   |       
|   \---__pycache__
|           custom_functions.cpython-311.pyc
|           
\---streamlit
    |   app.py
    |   requirements.txt
    |   
    +---images
    |       cover_image_sample.png
    |       custom_models_grad_cam_v1.png
    |       custom_model_history_v1.png
    |       custom_resnet50_grad_cam_v1.png
    |       data_exploration.png
    |       densenet.png
    |       densenet_gradcam_v1.png
    |       densenet_history_v1.png
    |       github.png
    |       linkedin.png
    |       logo.png
    |       model_comparison.png
    |       random_brain_image_sample.png
    |       resizing.png
    |       resnet50.jpg
    |       resnet_history_v1.png
    |       
    +---items
    |   |   comparison.py
    |   |   custom.py
    |   |   DenseNet162.py
    |   |   description.py
    |   |   exploration.py
    |   |   ResNet50.py
    |   |   
    |   \---__pycache__
    |           comparison.cpython-311.pyc
    |           custom.cpython-311.pyc
    |           DenseNet162.cpython-311.pyc
    |           description.cpython-311.pyc
    |           exploration.cpython-311.pyc
    |           ResNet50.cpython-311.pyc
    |           
    \---metrics
            custom_model_classification_report.pkl
            custom_model_scores.pkl
            custom_model_tumor_performance_v1.pkl
            densenet_classification_report.pkl
            densenet_scores.pkl
            densenet_tumor_performance_v1.pkl
            resnet50_tumor_performance_v1.pkl
            resnet_50_classification_report.pkl
            resnet_50_scores.pkl