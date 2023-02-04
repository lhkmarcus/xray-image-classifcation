import tensorflow as tf
from config.path_config import RAW_DATA_PATH

def get_dataset(
        directory=RAW_DATA_PATH,
        batch_size=32, 
        image_size=(300, 300), 
        subset=None):
    dataset = tf.keras.utils.image_dataset_from_directory(
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
    return dataset