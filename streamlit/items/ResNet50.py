import streamlit as st
from PIL import Image
import pandas as pd

def resnet50():

    with Image.open("images/resnet50.jpg") as img:
        st.image(img, caption = "CC BY-SA 4.0 license 2.3.4. MobileNet.",
                use_container_width = True)

    st.markdown("""<p class = 'scaling-headers'><u>ResNet50</u></p>""",
                unsafe_allow_html = True)

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

    with Image.open("images/resnet_history.png") as img:
        st.image(img, caption = """Training and Validation measurements.
                 Training data is in blue and validation data is in purple.
                 """, use_container_width = True)

    st.markdown("""The training process demonstrated a relatively high degree
                of instability, which was observed in the oscillating training
                ROC-AUC and F1-Scores. This may be due to the larger batch
                size (64) relative to the size of the training data. However,
                the validation set showed more stability, with more consistent
                increases and less fluctuation. The model triggered early
                stoppage, where the model was loaded from the best ROC-AUC
                score (0.9391) and was saved in that state. The model scored
                lower on the test set than on the validation set, with a
                ROC-AUC score of 0.868. The full classification report is
                printed below.""")

    resnet_report = {'Precision' : [0.80, 0.92, '', 0.86, 0.87],
                     'Recall' : [0.88, 0.86, '', 0.87, 0.87],
                     'F1-Score' : [0.83, 0.89, 0.87, 0.86, 0.87],
                     'Support' : [49, 78, 127, 127, 127]
                    }
    resnet_report = pd.DataFrame(resnet_report,
                                 index = ['No Tumor', 'Tumor','Accuracy',
                                          'Macro Average', 'Weighted Average'])

    resnet_report = resnet_report.style.format(precision = 2, na_rep = "")
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

    with Image.open("images/resnet_gradcam.png") as img:
        st.image(img, caption = """GradCAM images for twenty randomly selected
        images. The first two rows are 'No Tumor' and the last two rows
        are 'Tumor' images. Predicted label, with probability in brackets, and
        actual label are shown above each image.""", use_container_width = True)
                

    
