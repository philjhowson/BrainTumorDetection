import streamlit as st
from PIL import Image
import pandas as pd

def densenet162():

    with Image.open("images/densenet.png") as img:
        st.image(img)

    st.markdown("""<p class = 'scaling-headers'><u>DenseNet169</u></p>""",
                unsafe_allow_html = True)
    
    st.markdown("""I used DenseNet169 
                (<a href = "https://arxiv.org/abs/1608.06993">Huang et al., 2016</a>) in PyTorch 
                (<a href = "https://arxiv.org/abs/1912.01703">Paszke et al.,
                2019</a>). I froze all the pre-trained layers except the final four layers of
                the final (fourth) dense block. The final classifier hear was also 
                replaced with simplified linear layer with 1 output to facilitate classification 
                between the two categories of the dataset. This resulted in a total of 972,160 
                trainable parameters, out of a total 12,486,145 parameters. I used an Adam 
                optimizer, which was initialized with a learning rate of 1e-3 and weight decay 
                of 1e-4. As with the ResNet50 model, I scored the model performance 
                using the ROC-AUC and F1-Scores. The figure below shows the training history, 
                including the loss, the ROC-AUC, and the F1-Score for the training and 
                validation set, and the gradient for the training set""",
                unsafe_allow_html = True)
    
    with Image.open("images/densenet_history.png") as img:
        st.image(img, caption = """Training and Validation measurements.
                 Training data is in blue and validation data is in purple.
                 """, use_container_width = True)
        
    st.markdown("""The training process demonstrated a relatively high degree
                of instability, which was observed in the oscillating training
                ROC-AUC and F1-Scores. This was still present even with a smaller
                batch size (32) than was used for ResNet50. However,
                the validation set showed more stability, with more consistent
                increases and less fluctuation. The model was training was stopped
                early because it achieved a ROC-AUC of 1. The model scored
                lower on the test set than on the validation set, with a
                ROC-AUC score of 0.951. The full classification report is
                printed below.""")

    densenet_report = {'Precision' : [0.89, 0.99, '', 0.94, 0.95],
                     'Recall' : [0.98, 0.92, '', 0.95, 0.94],
                     'F1-Score' : [0.93, 0.95, 0.94, 0.94, 0.95],
                     'Support' : [49, 78, 127, 127, 127]
                    }
    densenet_report = pd.DataFrame(densenet_report,
                                 index = ['No Tumor', 'Tumor','Accuracy',
                                          'Macro Average', 'Weighted Average'])

    densenet_report = densenet_report.style.format(precision = 2, na_rep = "")
    st.table(densenet_report)

    st.markdown("""To further explore the how the model classifies scans as
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

    with Image.open("images/densenet_gradcam.png") as img:
        st.image(img, caption = """GradCAM images for twenty randomly selected
        images. The first two rows are 'No Tumor' and the last two rows
        are 'Tumor' images. Predicted label, with probability in brackets, and
        actual label are shown above each image.""", use_container_width = True)