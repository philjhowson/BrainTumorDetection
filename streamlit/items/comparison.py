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
