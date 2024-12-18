import streamlit as st
from items.description import project_description

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

</style>
"""
)


st.sidebar.image("images/logo.png", use_container_width = False)

menu = st.sidebar.radio("Menu", ["Poject Description",
                                 "Data Exploration",
                                 "ResNet50",
                                 "DenseNet162",
                                 "Custom Model",
                                 "Outlook"],
                        label_visibility = "collapsed")


if menu == "Poject Description":
    #project_description()
    pass
elif menu == "Data Exploration":
    #data_exploration()
    pass
elif menu == "ResNet50":
    #ResNet50()
    pass
elif menu == "DenseNet162":
    #DenseNet162()
    pass
elif menu == "Custom Model":
    #custom_model()
    pass
elif menu == "Outlook":
    #outlook()
    pass

