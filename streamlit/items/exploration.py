import streamlit as st
from PIL import Image

def exploration():
    st.markdown("""<p class = 'scaling-headers'><u>
                Data Exploration</u></p>""", unsafe_allow_html = True)

    st.markdown("""The dataset contains 253 images, of which, 155 (61%)
                of the images contained a positive tumor diagnosis. Image
                dimensions varied significantly as did the aspect ratios.
                The figure below presents the general characteristics of
                the image dataset.""")

    with Image.open("images/data_exploration.png") as img:
        st.image(img, caption = """Data Characteristics: (top left) Count for
                No ('No Tumor') and Yes ('Tumor'), (top middle) distribution
                of image widths, (top right) distribution of image heights,
                (bottom left) image aspect ratios, (bottom middle) count
                of square ('Yes') versus non-square ('No') image shapes, and
                (bottom right) image mode count.""")

    st.markdown("""The image dataset indicates that some preprocessing of
                the images will be necessary, not only to have a similar size
                across all images, but also to reshape the images to reduce
                the non-informative space. The figure below presents 10
                randomly selected images from the 'No' category and the 'Yes'
                category.""")

    with Image.open("images/random_brain_image_sample.png") as img:
        st.image(img, caption = """Twenty randomly selected images for
                the classes 'No Tumor' (top two rows) and 'Tumor' (bottom
                two rows).""")

    st.markdown("""<p class = 'scaling-headers'><u>
                Data Preparation</u></p>""", unsafe_allow_html = True)

    st.markdown("""As can be seen in the image, reshaping to a single set of
                dimensions across all images would result in significant distortion.
                Furthermore, many images contain a significant amount of black
                space. In order to deal with this, I wrote a custom reshaping function
                to minimize black space and maximize the image brain size without
                causing significant distortion. The figure below presents a sample
                image in it's original dimensions, using my reshaping function, and
                how the image would look if it were reshaped without the custom function.
                """)

    with Image.open("images/resizing.png") as img:
        st.image(img, caption = """(left) original image, (middle) reshaped
        image using my custom reshaping function, and (right) what the image
        would look like if it was reshaped to 400 x 400 using Image.resize()
        or another similar function.""")

    st.markdown("""DataLoader() was used to process batches for the training
                and validation stages. Batch sizes were set to 32 for ResNet50
                and DenseNet162, but to 16 for the custom model due to GPU limitations.
                I used the transforms.Compose() to create randomized changes
                in the images to reduce overfitting and help model generalization.
                They included vertical and horizontal flips, a random affine, gaussian
                blur, and random erasing. Additionally, the images were
                converted to grayscale, resized with my custom function,
                and normalized with a mean of 0.5, and std of 0.5. The
                training data included 80% of the data, and the remaining
                20% was split in half, 50% for a validation set and 50% for
                a test set. Class weights were computed with the
                compute_class_weight() function in sklearn
                (<a href = "https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf?ref=https:/">
                Pedregosa et al., 2011</a>) for use with a binary cross
                entropy with logits loss function to deal with class
                imbalances.""", unsafe_allow_html = True)
