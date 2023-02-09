import os
import warnings
import numpy as np

# import sys; sys.exit()

## Filter warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
print("\nUsing TensorFlow version " + tf.__version__)

from tensorflow import keras
# from src.models.vgg16 import vggModel
from tensorflow.python.keras import layers
from src.preliminaries.get_dataset import get_dataset
from src.preliminaries.get_dataset import train_valid_split
from src.models.config.metrics import f1_score, precision, recall
from src.models.config.callbacks import exp_decay_lr, early_stop, checkpoint

#### Create class trainModel
class trainModel(keras.Model):
    def __init__(self, model, weight_path, epochs, batch_size):
        super().__init__()
        self.model = model
        self.weight_path = weight_path
        self.epochs = epochs
        self.batch_size = batch_size

    def compile(self):
        print("\nLoading training and validation data...\n")
        self.train_ds = get_dataset()
        self.valid_ds = get_dataset(subset="validation")
        print("\nInstantiating model...")
        print("Compiling model...\n")
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy", f1_score, precision, recall]
        )
        print("--> Using sparse categorical crossentropy loss function.")
        print("--> Using Adam optimizer.")
        print("--> Using metrics: accuracy, F1_score, precision, and recall.\n")
        print("Printing model summary...\n")
        print(self.model.graph().summary())

    def fit(self):
        ckpt_path = os.path.join(self.weight_path, f"{self.model}", "cp-{epoch:04.d}.ckpt")
        print("\n--> Epochs set to {}.".format(self.epochs))
        print("--> Batch size set to {}.\n".format(self.batch_size))

        ## Fit model
        self.model_hist = self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            validation_data=self.valid_ds,
            callbacks=[exp_decay_lr, 
                       checkpoint(ckpt_path, self.batch_size), 
                       early_stop]
        )
    
    def save(self):
        self.model_history > "save"
