import streamlit as st
from PIL import Image
import pandas as pd
import pickle

def custom_model():

    st.markdown("""<h2 style="font-size: 1.5em;"><u>Custom Model</u></h2>""",
                unsafe_allow_html = True)

    with open('metrics/custom_model_scores_v2.pkl', 'rb') as f:
        scores = pickle.load(f)

    with open('metrics/custom_model_tumor_performance_v2.pkl', 'rb') as f:
        history = pickle.load(f)
    
    st.markdown("""I wrote a custom CNN with pytorch where the input images
                were sent through a large (kernel = 7) and medium (kernel = 5)
                before being concatinated and sent through the remainder of
                the network. Each block of convolutional layers was additionally
                fed the output feature matrix from each previous convolutional
                block in a similar manner to residual learning or densely
                connected networks from ResNet and DenseNet. The model had
                a total of 21,905,857 parameters, all of which were trainable.
                I used an Adam optimizer, which was initialized with a learning rate
                of 1e-3 and weight decayof 1e-4. As with the ResNet50 model, I
                scored the model performance using the ROC-AUC and F1-Scores.
                The figure below shows the training history, including the loss,
                the ROC-AUC, and the F1-Score for the training and 
                validation set, and the gradient for the training set
                """, unsafe_allow_html = True)

    with Image.open("images/custom_model_history_v2.png") as img:
        st.image(img, caption = """Training and Validation measurements.
                 Training data is in blue and validation data is in purple.
                 """, use_container_width = True)

    st.markdown(f"""The training and validation data revealed a high amount of
                fluctuation, although there was a general overall trend upwards. The
                validation metrics were typically above the training metrics,
                indicating good generalization. The model training was stopped
                early and the model was loaded from the best ROC-AUC score 
                ({round(max(history['val_roc_auc']), 2)}). The model scored
                lower on the test set than on the validation set, with a
                ROC-AUC score of {round(scores['ROC-AUC'], 2)}. The full
                classification report is printed below.""")

    with open(f"metrics/custom_model_classification_report_v2.pkl", 'rb') as f:
              custom_report = pickle.load(f)

    accuracy = custom_report.pop('accuracy')
    custom_report = pd.DataFrame(custom_report).transpose()
    accuracy_row = pd.DataFrame({'precision': '', 'recall': '', 'f1-score': accuracy, 'support': 127}, index=['accuracy'])
    custom_report = pd.concat([custom_report.iloc[:2, :], accuracy_row, custom_report.iloc[2:, :]])
    custom_report['support'] = custom_report['support'].astype(int)
    custom_report.rename(columns = {'precision' : 'Precision', 'recall' : 'Recall',
                                    'f1-score' : 'F1-Score', 'support' : 'Support'}, inplace = True)
    custom_report = custom_report.style.format(precision = 3, na_rep = "")

    st.table(custom_report)   

    st.markdown("""To further explore how the model classifies scans as
                either 'No Tumor' or 'Tumor', I used GradCAM from the torchcam
                (<a href = "https://github.com/frgfm/torch-cam">Fernandez,
                2020</a>) library. In most cases, where the label was 'Tumor'
                and the model predicted 'Tumor', activations were strongest
                on or around the tumor, suggesting the model has identified
                what a tumor looks like in an MRI image. In most classifications
                as 'No Tumor', the model had very little activation. In fact,
                for some images  there was no registered activation patters.
                This is likely because of the binary nature of the classification
                task. When the model is not predicting the target class, there
                is little activation. The minimal activations in other 'No Tumor'
                cases and the rich activation in the 'Tumor' cases seem to confirm
                this. In images with 'No Tumor' and visible activations, activation
                was often (but not always) centralized on the brain matter,
                suggesting the clues to 'No Tumor' vs. 'Tumor' are present in
                various structors of the brain.""",
                unsafe_allow_html = True)

    with Image.open("images/custom_model_grad_cam_v2.png") as img:
        st.image(img, caption = """GradCAM images for twenty randomly selected
        images. The first two rows are 'No Tumor' and the last two rows
        are 'Tumor' images. Predicted label, with probability in brackets, and
        actual label are shown above each image.""", use_container_width = True)
