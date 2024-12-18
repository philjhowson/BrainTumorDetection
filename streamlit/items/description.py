import streamlit as st
from PIL import Image

def description():
    
    with Image.open('images/cover_image_sample.png') as img:
        st.image(img)

    st.markdown(
    """
    <style>
    .scaling-headers {
        font-size: 1.75vw;
        #text-align: center;
    }
    </style>
    """,
    unsafe_allow_html = True)

    st.markdown("""<p class = 'scaling-headers'><u>Machine
                Learning Models for Brain Tumor Detection
                </u></p>""",
                unsafe_allow_html = True)

    st.markdown("""
        The objective of this project is to utilize cutting edge machine 
        learning technology to train models to detect brain tumors based 
        on MRI images. In this project, I make use of three different models,
        ResNet50, DenseNet, and a custom made model to assess their ability
        to detect brain tumors in MRI images.
        """)

    st.markdown("""<p class = 'scaling-headers'><u>The Dataset</u></p>""",
                unsafe_allow_html = True)
    
    st.markdown("""Dataset retrieved from kaggle:<br>
        <a href = "https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data">
        Brain MRI Images for Brain Tumor Detection</a><br>
        This dataset contrains MRI images of brains with and without tumors.
        The objective of this project is to build a neural network that can
        detect the presence of a brain tumor based on an MRI scans. To achieve
        this objective, I trained three different models to evaluate
        performance.""", unsafe_allow_html = True)

    st.markdown("""For access to the files and code associated with this project, please
        visit my <a href = "https://github.com/philjhowson/BrainTumorDetection">GitHub</a>.
        If you want access to the dataset, please click on the link to the dataset on
        kaggle provided above.""", unsafe_allow_html = True)
