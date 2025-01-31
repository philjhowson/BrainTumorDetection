import streamlit as st
from PIL import Image

def comparison():

    st.markdown("""<p class = 'scaling-headers'><u>Model Comparison</u></p>""",
                unsafe_allow_html = True)

    st.markdown("""To compare the overall performance on the test data
                for each of the models, I used the roc_auc_score() and
                f1_score() functions from the sklearn.metrics library.
                A barplot that presents the results of the comparison
                is presented below.
                """, unsafe_allow_html = True)

    with Image.open("images/model_comparison.png") as img:
        st.image(img, caption = """ROC-AUC and F1 Scores for each of the
                 models. The ROC-AUC score is in blue and the F1-Score
                 is in purple.
                 """, use_container_width = True)

    st.markdown("""Overall, there were similar levels of performance for
                each of the models, although the custom model scored
                noticably lower on the ROC-AUC metric. This may be in
                part due to the fact that both Resnet and DenseNet were
                trained on a large set of images prior to being trained
                on the MRI images. This was also reflected in the fact that
                the custom model completed significantly more epochs before
                early stopping was triggered. With further improvements
                to the model architecture and finer hyper parameter tuning,
                a better score could likely be achieved. However, the model
                takes much longer to train because it requires more
                trainable parameters and because it has to be trained
                directly from weight initialization and cannot benefit from
                transfer learning.
                """)

    st.markdown("""DenseNet was the fastest model to train, partially due
                to the smaller number of parameters, both trainable and
                overall, but also due to the fact that it was able to
                benefit from transfer learning. The model is also significantly
                more lightweight compared to ResNet (49 MB vs. 359 MB), and
                while ResNet had the same advantages with transfer learning,
                DenseNet still outperformed it. With more hyperparameter
                tuning, it could also likely perform even better and within
                a short period of time.""")
