from torchvision import datasets
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from custom_functions import custom_resize

"""
here I load in the dataset without transformations so I can further explore the original dataset to see
what transformations make the most sense
"""

dataset = datasets.ImageFolder(root = 'scans')

print(dataset)
print(dataset.classes)

width = []
height = []
category = []
mode = []

for item in dataset.samples:
    with Image.open(item[0]) as img:
        width.append(img.width)
        height.append(img.height)
        category.append(item[1])
        mode.append(img.mode)

data = pd.DataFrame({'Category' : category, 'Width' : width, 'Height' : height, 'Mode' : mode})

data['Aspect_Ratio'] = data['Width'] / data['Height']
data['Shapes'] = [1 if data['Width'].iloc[item] == data['Height'].iloc[item] else 0 for item in range(len(data))]

data.describe()

fig, ax = plt.subplots(2, 3, figsize = (20, 10))

sns.countplot(data = data, x = 'Category', ax = ax[0, 0])
ax[0, 0].set_xticks(ticks = [0, 1], labels = ['No', 'Yes'])
ax[0, 0].set_title('Countplot for No/Yes Categories')

sns.histplot(data = data, x = 'Width', kde = True, ax = ax[0, 1])
ax[0, 1].set_title('Histogram of Image Widths');

sns.histplot(data = data, x = 'Height', kde = True, ax = ax[0, 2])
ax[0, 2].set_title('Histogram of Image Heights');

sns.histplot(data = data, x = 'Aspect_Ratio', kde = True, ax = ax[1, 0])
ax[1, 0].set_title('Histogram of Aspect Ratios');

sns.countplot(data = data, x = 'Shapes', ax = ax[1, 1])
ax[1, 1].set_xticks(ticks = [0, 1], labels = ['No', 'Yes'])
ax[1, 1].set_title('Countplot for Shape (Yes = Square)')

sns.countplot(data = data, x = 'Mode', ax = ax[1, 2])
ax[1, 2].set_title('Image Mode');

plt.tight_layout();

plt.savefig('images/data_exploration.png', bbox_inches = 'tight')

no_set = dataset.samples[:98]
yes_set = dataset.samples[98:]

items = [48, 55,  2, 72, 53, 12, 64, 83, 82, 21,  3,  5, 79, 32, 59, 81, 33, 58,  7, 20]

fig, ax = plt.subplots(4, 5, figsize = (20, 15))

for index, axes in enumerate(ax.flat):

    if index < 10:

        with Image.open(no_set[items[index]][0]) as img:
            axes.imshow(img, cmap = 'gray')
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_title('No Tumor')

    else:
        with Image.open(yes_set[items[index]][0]) as img:
            axes.imshow(img, cmap='gray')
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_title('Tumor')

plt.tight_layout();
plt.savefig('random_brain_image_sample.png', bbox_inches = 'tight')

img1 = Image.open(no_set[items[2]][0])
img1 = img1.convert('L')
img2 = custom_resize(img1)

img3 = img1.resize((400, 400), Image.LANCZOS)

titles = ['Original Image:', 'Reshaped Image:', 'Untrimmed Reshaped Image:']
images = [img1, img2, img3]

fig, ax = plt.subplots(1, 3, figsize = (15, 5))

for index, axes in enumerate(ax):
    axes.imshow(images[index], cmap = 'gray')
    axes.set_title(f"{titles[index]} {images[index].width} x {images[index].height}")
    axes.set_xticks([])
    axes.set_yticks([])

plt.savefig('images/resizing.png', bbox_inches = 'tight')
