import os
import warnings
import numpy as np

## Filter warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from config.get_dataset import get_dataset
from config.model.vgg_16 import vggModel
from config.path_config import MODEL_WEIGHTS, RAW_DATA_PATH
from config.model.eval_metrics import f1_score, precision, recall
from config.model.callbacks import exp_decay_lr, early_stop, checkpoint 

print("--------------------------------------------------------------")
print("Using TensorFlow version " + tf.__version__)
print("--------------------------------------------------------------")

print("--------------------------------------------------------------")
print("LOADING TRAINING AND VALIDATION DATA...\n")
training_ds = get_dataset(directory=RAW_DATA_PATH, subset="training")
validation_ds = get_dataset(directory=RAW_DATA_PATH, subset="validation")
print("--------------------------------------------------------------")

print("--------------------------------------------------------------")
print("INSTANTIATING MODEL...")
model = vggModel()
print("COMPILING MODEL...\n")
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy", f1_score, precision, recall]
)
print("--> Using Sparse Categorical Crossentropy loss function.")
print("--> Using Adam optimizer.")
print("--> Using metrics: accuracy, F1_score, precision, and recall.\n")
print("PRINTING MODEL SUMMARY...")
print(model.graph().summary())
print("--------------------------------------------------------------")
epochs = 50
batch_size = 32
checkpoint_path = os.path.join(MODEL_WEIGHTS, "cp-{epoch:04.d}.ckpt")

print("--> Epochs set to {}.".format(epochs))
print("--> Batch size set to {}.".format(batch_size))
print("--------------------------------------------------------------")

## Fit model
# model_history = model.fit(
#     training_ds,
#     epochs=epochs,
#     validation_data=validation_ds,
#     callbacks=[exp_decay_lr, checkpoint(checkpoint_path, batch_size), early_stop]
# )
