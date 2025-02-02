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

    with Image.open("images/model_comparison_2_2_2.png") as img:
        st.image(img, caption = """ROC-AUC and F1 Scores for each of the
                 models. The ROC-AUC score is in blue and the F1-Score
                 is in purple.
                 """, use_container_width = True)

    st.markdown("""Overall, there were similar levels of performance for
                ResNet50 and DenseNet162. The custom model underperformed
                compared to the other two with lower ROC-AUC and F1-scores.
                Further optimization of the model architecture and training
                parameters would likely facilitate a more competative model.
                The custom model also had significantly lower parameter
                count than ResNet50 (but similar parameter count as
                Densenet162), which may contribute to lower performance.
                Additionally, the other models had the benefit of transfer
                learning from a robust dataset of images.
                """)

    st.markdown("""Nonetheless, DenseNet was the fastest model to train,
                partially due to the smaller number of parameters, both
                trainable and overall, but also due to the fact that it was able to
                benefit from transfer learning. The model is also significantly
                more lightweight compared to ResNet (49 MB vs. 359 MB), and
                while ResNet had the same advantages with transfer learning,
                DenseNet still outperformed it. With more hyperparameter
                tuning, it could also likely perform even better and within
                a short period of time.""")

    st.markdown("""All of the models likely suffered from the small database
                that was available for usage. In order to make monumental leaps
                in accuracy, much larger datasets are needed. The societal
                benefits could leap to both earlier and more accurate
                diagnosis and has the potential to save many lives.""")
