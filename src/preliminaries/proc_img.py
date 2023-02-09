import os
import warnings

## Filter warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
import sort_img
import tensorflow as tf
import matplotlib.pyplot as plt

## Sort images
sort_img.main(
    parent_folder="./data/raw/",
    extract=False,
    sort=True,
    drop_parent=False
)

## Load images with a 32 batch size
print("--------------------------------------------------------------")
print("Loading images from " + RAW_DATA_PATH + "...")
print("--------------------------------------------------------------")
images = tf.keras.utils.image_dataset_from_directory(
    directory=RAW_DATA_PATH,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=32,
    image_size=(300, 300),
    shuffle=True
)
print("Images loaded.")
print("--------------------------------------------------------------")

## Get class names
class_names = images.class_names

## Configure plot parameters
plt.style.use("seaborn-whitegrid")
matplotlib.rcParams["figure.figsize"] = (12, 9)
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

## Display images
plt.figure(figsize=(10, 10))
for images, labels in images.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        ax.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()

## Preprocess images
## Rescale, configure contrast, exposure, zoom, shear, and aspect ratio (hard)
## Set up pipeline

## Save images to processed folder
