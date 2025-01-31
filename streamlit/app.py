import streamlit as st
from items.description import description
from items.exploration import exploration
from items.ResNet50 import resnet50
from items.DenseNet162 import densenet162
from items.custom import custom_model
from items.comparison import comparison

st.html(
    """
<style>
[data-testid="stSidebarContent"] {
    background: white;
    /* Gradient background */
    color: white; /* Text color */
    padding: 5px; /* Add padding */
}

/* Main content area */
[data-testid="stAppViewContainer"] {
    background: white;
    padding: 5px; /* Add padding for the main content */
    border-radius: 5px; /* Add rounded corners */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
}

/* Apply Times New Roman font style globally */
body {
    font-family: 'Roboto', sans-serif;
    font-size: 16px; /* Set the font size */
    color: black; /* Set text color */   
}

/* style other elements globally */
h1, h2, h3 {
    font-family: 'Roboto', sans-serif;
    color: black; /* Set a color for headers */
    width: 100% !important;
}

/* Customize the sidebar text */
[data-testid="stSidebarContent"] {
    font-family: 'Roboto', sans-serif;
    color: black;
}

/* Change the text color of the entire sidebar */
[data-testid="stSidebar"] {
    color: black !important;
}

/* Change the color of the radio button labels */
.stRadio label {
    color: black !important;
}

/* Change the color of the radio button option text */
.stRadio div {
    color: black !important;
}

/* Change the text color for the entire main content area */
body {
    color: black !important;
}

/* Change the color of text in markdown and other text elements */
.stMarkdown, .stText {
    color: black !important;
}

/* Adjust the width of the main content area */
div.main > div {
    width: 80% !important;
    margin: 0 auto;  /* Center the content */
}


.scaling-headers {
    font-size: 1.75vw;
    #text-align: center;
}

</style>
"""
)


st.sidebar.image("images/logo.png", use_container_width = False)

menu = st.sidebar.radio("Menu", ["Project Description",
                                 "Data Exploration & Preparation",
                                 "ResNet50",
                                 "DenseNet162",
                                 "Custom Model",
                                 "Model Comparison"],
                        label_visibility = "collapsed")


if menu == "Project Description":
    description()
    
elif menu == "Data Exploration & Preparation":
    exploration()

elif menu == "ResNet50":
    resnet50()

elif menu == "DenseNet162":
    densenet162()

elif menu == "Custom Model":
    custom_model()

elif menu == "Model Comparison":
    comparison()
    pass

col1, col2 = st.sidebar.columns([0.15, 0.85], gap = "small",
                                vertical_alignment = "center")
col1.image("images/github.png")
col2.markdown("""<a href = "https://philjhowson.github.io">philjhowson.github.io</a>""",
              unsafe_allow_html = True)

col1, col2 = st.sidebar.columns([0.15, 0.85], gap = "small",
                                vertical_alignment = "center")
col1.image("images/linkedin.png")
col2.markdown("""<a href = "https://www.linkedin.com/in/philjhowson/">philjhowson</a>""",
              unsafe_allow_html = True)

