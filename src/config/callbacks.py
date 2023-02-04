import math
import tensorflow as tf
from tensorflow import keras

## Exponential decay learning rate callback
def exp_decay(initial_learning_rate, epoch):
    k = 0.1
    return initial_learning_rate * math.exp(-k * epoch)

exp_decay_lr = tf.keras.callbacks.LearningRateScheduler(exp_decay, verbose=1)

## Checkpoint callback
def checkpoint(path, batch_size):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        verbose=1,
        save_weights_only=True,
        save_freq=batch_size * 5,
    )
    return checkpoint

## Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    verbose=1,
    restore_best_weights=True
)