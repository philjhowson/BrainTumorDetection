import streamlit as st
from PIL import Image
import pandas as pd
import pickle

def densenet162():

    with Image.open("streamlit/images/densenet.png") as img:
        st.image(img, caption = "Image credit to Huang et al., 2018.")

    st.markdown("""<h2 style="font-size: 1.5em;"><u>DenseNet162</u></h2>""",
                unsafe_allow_html = True)

    with open('metrics/densenet_scores_v2.pkl', 'rb') as f:
        scores = pickle.load(f)

    with open('metrics/densenet_tumor_performance_v2.pkl', 'rb') as f:
        history = pickle.load(f)
    
    st.markdown("""I used DenseNet162 
                (<a href = "https://arxiv.org/abs/1608.06993">Huang et al., 2016</a>) in PyTorch 
                (<a href = "https://arxiv.org/abs/1912.01703">Paszke et al.,
                2019</a>). I froze all the pre-trained layers except the final four layers of
                the final (fourth) dense block. The final classifier was also 
                replaced with simplified linear layer with 1 output to facilitate classification 
                between the two categories of the dataset. This resulted in a total of 972,160 
                trainable parameters, out of a total 12,486,145 parameters. I used an Adam 
                optimizer, which was initialized with a learning rate of 1e-3 and weight decay 
                of 1e-4. As with the ResNet50 model, I scored the model performance 
                using the ROC-AUC and F1-Scores. The figure below shows the training history, 
                including the loss, the ROC-AUC, and the F1-Score for the training and 
                validation set, and the gradient for the training set""",
                unsafe_allow_html = True)
    
    with Image.open("images/densenet_history_v2.png") as img:
        st.image(img, caption = """Training and Validation measurements.
                 Training data is in blue and validation data is in purple.
                 """, use_container_width = True)
        
    st.markdown(f"""The training and validation metrics indicated a relatively stable
                increase in metrics over the training epochs, although there were
                still choppy increases and decreases in the performance metrics.
                Additionally, the best score was approached fairly early, with only
                small increases in scores and decreases in loss over the remaining
                epochs. The model training was stopped early and the model
                was loaded from the best ROC-AUC score 
                ({round(max(history['val_roc_auc']), 2)}). The model scored
                lower on the test set than on the validation set, with a
                ROC-AUC score of {round(scores['ROC-AUC'], 2)}. The full
                classification report is printed below.""")

    with open(f"metrics/densenet_classification_report_v2.pkl", 'rb') as f:
              densenet_report = pickle.load(f)

    accuracy = densenet_report.pop('accuracy')
    densenet_report = pd.DataFrame(densenet_report).transpose()
    accuracy_row = pd.DataFrame({'precision': '', 'recall': '', 'f1-score': accuracy, 'support': 127}, index=['accuracy'])
    densenet_report = pd.concat([densenet_report.iloc[:2, :], accuracy_row, densenet_report.iloc[2:, :]])
    densenet_report['support'] = densenet_report['support'].astype(int)
    densenet_report.rename(columns = {'precision' : 'Precision', 'recall' : 'Recall',
                                    'f1-score' : 'F1-Score', 'support' : 'Support'}, inplace = True)
    densenet_report.index = ['No Tumor', 'Tumor', 'Accuracy', 'Macro Average', 'Weighted Average']
    densenet_report = densenet_report.style.format(precision = 3, na_rep = "")

    st.table(densenet_report)

    st.markdown("""To further explore how the model classifies scans as
                either 'No Tumor' or 'Tumor', I used GradCAM from the torchcam
                (<a href = "https://github.com/frgfm/torch-cam">Fernandez,
                2020</a>) library. In most cases where the label was 'Tumor'
                and the model predicted 'Tumor', activations were strongest
                over the entirety of the brain matter, suggesting the model 
                utilizes the information from the entire brain image to classify 
                an image as 'Tumor'. In most classifications as 'No Tumor', 
                the model had very little activation. In fact, for some images 
                there was no registered activation patters. This is likely because 
                of the binary nature of the classification task. When the model 
                is not predicting the target class, there is little activation. 
                The minimal activations in other 'No Tumor' cases and the rich 
                activation in the 'Tumor' cases seem to confirm this.""",
                unsafe_allow_html = True)

    with Image.open("images/densenet_grad_cam_v2.png") as img:
        st.image(img, caption = """GradCAM images for twenty randomly selected
        images. The first two rows are 'No Tumor' and the last two rows
        are 'Tumor' images. Predicted label, with probability in brackets, and
        actual label are shown above each image.""", use_container_width = True)
