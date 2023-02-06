import os
import warnings
import numpy as np

## Filter warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

print("--------------------------------------------------------------")
print("Using TensorFlow version " + tf.__version__)
print("--------------------------------------------------------------")

from tensorflow import keras

from src.preliminaries import get_dataset
from src.models.vgg16 import vggModel
from src.config_paths import MODEL_WEIGHTS, RAW_DATA_PATH
from src.models.config.metrics import f1_score, precision, recall
from src.models.config.callbacks import exp_decay_lr, early_stop, checkpoint

class Model(keras.Model):
    def __init__(self, model, epochs, batch_size):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def compile(self):
        print("--------------------------------------------------------------")
        print("Loading training and validation data...\n")
        self.training_ds = get_dataset(directory=RAW_DATA_PATH, subset="training")
        self.validation_ds = get_dataset(directory=RAW_DATA_PATH, subset="validation")
        print("--------------------------------------------------------------")
        print("--------------------------------------------------------------")
        print("Instantiating model...")
        print("Compiling model...\n")
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy", f1_score, precision, recall]
        )
        print("--> Using Sparse Categorical Crossentropy loss function.")
        print("--> Using Adam optimizer.")
        print("--> Using metrics: accuracy, F1_score, precision, and recall.\n")
        print("Printing model summary...")
        print(self.model.graph().summary())
        print("--------------------------------------------------------------")

    def fit(self):
        checkpoint_path = os.path.join(MODEL_WEIGHTS, "cp-{epoch:04.d}.ckpt")
        print("--> Epochs set to {}.".format(self.epochs))
        print("--> Batch size set to {}.".format(self.batch_size))
        print("--------------------------------------------------------------")

        ## Fit model
        ## Save history to ./history
        self.model_history = self.model.fit(
            self.training_ds,
            epochs=self.epochs,
            validation_data=self.validation_ds,
            callbacks=[exp_decay_lr, 
                       checkpoint(checkpoint_path, self.batch_size), 
                       early_stop]
        )
    
    def save(self):
        self.model_history > "save"

if __name__ == "__main__":
    model = Model(
        model=vggModel,
        epochs=50,
        batch_size=32)
    model.compile()
    model.fit()
    model.save()