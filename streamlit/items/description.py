import streamlit as st
from PIL import Image

def description():
    
    with Image.open('images/cover_image_sample.png') as img:
        st.image(img)

    st.markdown("""<p class = 'scaling-headers'><u>Machine
                Learning Models for Brain Tumor Detection
                </u></p>""",
                unsafe_allow_html = True)

    st.markdown("""
        The objective of this project is to utilize cutting edge machine 
        learning technology to train models to detect brain tumors based 
        on MRI images. In this project, I make use of three different models,
        ResNet50, DenseNet169, and a custom made model to assess their ability
        to detect brain tumors in MRI images.
        """)

    st.markdown("""<p class = 'scaling-headers'><u>The Dataset</u></p>""",
                unsafe_allow_html = True)
    
    st.markdown("""Dataset retrieved from kaggle:<br>
        <a href = "https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data">
        Brain MRI Images for Brain Tumor Detection</a><br>
        This dataset contains MRI images of brains with and without tumors.""", unsafe_allow_html = True)

    st.markdown("""<p class = 'scaling-headers'><u>The Models</u></p>""",
                unsafe_allow_html = True)

    st.markdown("""
        <b>ResNet 50</b> (<a href = "https://arxiv.org/abs/1512.03385">He et al., 2015</a>)<br>
        Resnet50 utilizes residual learning through "skip connections" to facilitate deeper neural
        networks without loss of accuracy which occurred in earlier models.
        """, unsafe_allow_html = True)

    st.markdown("""
        <b>DenseNet 162</b> (<a href = "https://arxiv.org/abs/1608.06993">Huang et al., 2018</a>)<br>
        DenseNet169 differs from other models in that feature maps from all previous layers of the
        model are shared with each layer. This interconnectedness facilitates deeper neural networks
        with less computational load than other models.
        """, unsafe_allow_html = True)

    st.markdown("""
        <b>Custom Model</b></br>
        I built a custom model that utilizes different kernel sizes to leverage their ability to
        capture local and global relationships before concatenating the feature maps for pass through
        deeper convolutional layers. Each block in the network shares the output feature map of the
        previous blocks to facilitate a deeper neural network much like ResNet 50 and DenseNet 162.
        """, unsafe_allow_html = True)
    
    st.markdown("""For access to the files and code associated with this project, please
        visit my <a href = "https://github.com/philjhowson/BrainTumorDetection">GitHub</a>.
        If you want access to the dataset, please click on the link to the dataset on
        kaggle provided above.""", unsafe_allow_html = True)
