import tensorflow as tf

IMG_DIR = "..\\data\\images\\chest_xray"
BATCH_SIZE = 32
IMAGE_SIZE = (300, 300)

def get_dataset(directory=IMG_DIR, batch_size=32, image_size=(300, 300), subset="training"):
    return tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=42,
        validation_split=0.1,
        subset=subset,
    )

def train_valid_split():
    train_ds = get_dataset()
    valid_ds = get_dataset(subset="validation")
    return (train_ds, valid_ds)