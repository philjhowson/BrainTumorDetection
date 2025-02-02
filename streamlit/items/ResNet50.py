import streamlit as st
from PIL import Image
import pandas as pd
import pickle

def resnet50():

    with Image.open("streamlit/images/resnet50.jpg") as img:
        st.image(img, caption = "CC BY-SA 4.0 license 2.3.4. MobileNet.",
                use_container_width = True)

    st.markdown("""<h2 style="font-size: 1.5em;"><u>ResNet50</u></h2>""",
                unsafe_allow_html = True)

    with open('metrics/resnet50_scores_v2.pkl', 'rb') as f:
        scores = pickle.load(f)

    with open('metrics/resnet50_tumor_performance_v2.pkl', 'rb') as f:
        history = pickle.load(f)

    st.markdown("""I used a ResNet50
                (<a href = "https://arxiv.org/abs/1512.03385">He et al.,
                2015</a>) backbone in PyTorch 
                (<a href = "https://arxiv.org/abs/1912.01703">Paszke et al.,
                2019</a>) and removed the fully connected layers. I
                added four convolutional layers with residual connections,
                followed by two fully connected layers, with batch
                normalization and dropout, before the finally classifier
                layer. The purpose of the additional layers was to
                train them for task specific feature recognition. The
                additional convolutional layers were initialized with
                Kaiming normal initialization and the fully connected layers
                were initialized with Xavier uniform initialization. Initial
                layers of the ResNet50 backbone were frozen in order to
                preserve the learned features and facilitate transfer
                learning. The model had a total of 91,838,657 parameters, 
                of which 61,517,569 parameters were trainable.
                I used an Adam optimizer, which was initialized with a
                learning rate of 1e-3 and a weight decay of 1e-3. The training
                and validation set were scored using the ROC-AUC score and
                F1-Score due to the imbalanced nature of the dataset. The
                figure below shows the training history, including the loss,
                the ROC-AUC, and the F1-Score for the training and validation
                set, and the gradient for the training set.""",
                unsafe_allow_html = True)

    with Image.open("images/resnet50_history_v2.png") as img:
        st.image(img, caption = """Training and Validation measurements.
                 Training data is in blue and validation data is in purple.
                 """, use_container_width = True)

    st.markdown(f"""The training process demonstrated a relatively high degree
                of instability, which was observed in the oscillating training
                ROC-AUC and F1-Scores. This may be due to the larger batch
                size (32) relative to the size of the training data. The validation
                set showed the same instability, but with more increases in scores.
                The model triggered early stoppage, where the model was loaded
                from the best ROC-AUC score ({round(max(history['val_roc_auc']), 2)})
                and was saved in that state. The model scored lower on the
                test set than on the validation set, with a ROC-AUC score of
                {round(scores['ROC-AUC'], 2)}. The full classification report is
                printed below.""")

    with open(f"metrics/resnet50_classification_report_v2.pkl", 'rb') as f:
              resnet_report = pickle.load(f)

    accuracy = resnet_report.pop('accuracy')
    resnet_report = pd.DataFrame(resnet_report).transpose()
    accuracy_row = pd.DataFrame({'precision': '', 'recall': '', 'f1-score': accuracy, 'support': 127}, index=['accuracy'])
    resnet_report = pd.concat([resnet_report.iloc[:2, :], accuracy_row, resnet_report.iloc[2:, :]])
    resnet_report['support'] = resnet_report['support'].astype(int)
    resnet_report.rename(columns = {'precision' : 'Precision', 'recall' : 'Recall',
                                    'f1-score' : 'F1-Score', 'support' : 'Support'}, inplace = True)
    resnet_report = resnet_report.style.format(precision = 3, na_rep = "")

    st.table(resnet_report)

    st.markdown("""To further explore the how the model classifies scans as
                either 'No Tumor' or 'Tumor', I used GradCAM from the torchcam
                (<a href = "https://github.com/frgfm/torch-cam">Fernandez,
                2020</a>) library. In most cases where the label was 'Tumor'
                and the model predicted 'Tumor', activations were strongest
                on the tumor, suggesting the model has good recognition of
                the physical appearances of a tumor. In most classifications
                as 'No Tumor', activations focused on the outer areas and
                skull suggesting that no learned features for tumor recognition
                were found in the brain matter.""", unsafe_allow_html = True)

    with Image.open("images/resnet50_grad_cam_v2.png") as img:
        st.image(img, caption = """GradCAM images for twenty randomly selected
        images. The first two rows are 'No Tumor' and the last two rows
        are 'Tumor' images. Predicted label, with probability in brackets, and
        actual label are shown above each image.""", use_container_width = True)
                

    
