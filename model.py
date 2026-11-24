# CNN model definition for MNIST
# 3 conv blocks → flatten → dense → softmax

import tensorflow as tf
from tensorflow.keras import layers, Sequential


def build_cnn_model():
    """
    builds and compiles the CNN model

    3 convolutional layers with increasing filters (32 → 64 → 128),
    each followed by max pooling to shrink the spatial size.
    then flatten, one dense layer with dropout, and softmax output for 10 digits.
    """
    model = Sequential([
        # first conv block - 32 filters
        # takes in 28x28x1 grayscale images
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            input_shape=(28, 28, 1),
            name='conv1'
        ),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        # now 14x14x32

        # second conv block - 64 filters
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2'
        ),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        # now 7x7x64

        # third conv block - 128 filters
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv3'
        ),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),
        # now 3x3x128

        # flatten into a 1D vector so we can feed it into dense layers
        layers.Flatten(name='flatten'),

        # fully connected layer
        layers.Dense(256, activation='relu', name='dense1'),
        layers.Dropout(0.3, name='dropout1'),  # helps prevent overfitting

        # output layer - 10 neurons, one per digit
        layers.Dense(10, activation='softmax', name='output')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_cnn_model_summary():
    """just builds the model and prints summary, handy for checking the architecture"""
    model = build_cnn_model()
    model.summary()
    return model


if __name__ == "__main__":
    build_cnn_model_summary()