# handles loading the MNIST data, normalizing it, splitting it,
# and setting up the augmentation pipeline

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def load_data():
    """downloads and returns the MNIST dataset (happens automatically first time)"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, y_train, X_test, y_test, validation_split=0.1):
    """
    gets the data ready for the CNN:
    - normalize pixels from 0-255 to 0-1 (makes training more stable)
    - reshape to add channel dim since Conv2D expects (height, width, channels)
    - carve out a validation set from the training data
    """
    # normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # add the channel dimension
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # split off validation data from training set
    val_size = int(len(X_train) * validation_split)
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# keeping this at module level so tf.data.map doesnt try to
# recreate it on every call (causes errors with tf.function)
_DATA_AUGMENTATION = None


def get_data_augmentation():
    """
    sets up random transforms for training images.
    small rotations, translations, and zoom so the model doesnt just
    memorize exact pixel positions.

    no flipping though - flipping a 6 makes it look like a 9 which
    would mess up the labels
    """
    global _DATA_AUGMENTATION

    if _DATA_AUGMENTATION is None:
        _DATA_AUGMENTATION = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(factor=(-0.1, 0.1)),
            tf.keras.layers.RandomTranslation(
                height_factor=(-0.1, 0.1),
                width_factor=(-0.1, 0.1)
            ),
            tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1)),
        ], name='data_augmentation')

    return _DATA_AUGMENTATION


def reset_data_augmentation():
    """resets the augmentation layer, mainly here for testing"""
    global _DATA_AUGMENTATION
    _DATA_AUGMENTATION = None


def prepare_full_training_data(X_train, y_train, X_val, y_val):
    """
    wraps everything into tf.data pipelines.
    training data gets shuffled and augmented on the fly.
    validation data just gets batched, no augmentation on that.
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # shuffle + batch + augment for training
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(64)
    train_dataset = train_dataset.map(
        lambda x, y: (get_data_augmentation()(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    # just batch for validation, no augmentation
    val_dataset = val_dataset.batch(64)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


def prepare_test_data(X_test, y_test):
    """simple batched dataset for test evaluation"""
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(64)
    return test_dataset