# training script for the MNIST CNN
# loads data, builds the model, trains it with augmentation + early stopping,
# and saves the best model to disk

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocessing import load_data, preprocess_data, prepare_full_training_data
from model import build_cnn_model


MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'mnist_model.keras')
HISTORY_PATH = os.path.join(MODEL_DIR, 'history.npy')

MAX_EPOCHS = 30
MIN_EPOCHS = 10  # dont let early stopping kick in before this


def train_model():
    """
    runs the full training pipeline:
    load data → preprocess → build model → train → save
    """
    print("=" * 50)
    print("MNIST Digit Recognition - CNN Training")
    print("=" * 50)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # step 1: grab the data
    print("\n[1/5] Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_data()

    # step 2: normalize and split
    print("\n[2/5] Preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test, validation_split=0.1
    )

    # step 3: set up data pipelines with augmentation
    print("\n[3/5] Preparing data pipelines...")
    train_dataset, val_dataset = prepare_full_training_data(
        X_train, y_train, X_val, y_val
    )

    # step 4: build the CNN
    print("\n[4/5] Building CNN model...")
    model = build_cnn_model()
    model.summary()

    # step 5: train
    print("\n[5/5] Training model...")
    print(f"Max epochs: {MAX_EPOCHS}, Min epochs: {MIN_EPOCHS}")

    callbacks = [
        # stop early if val_loss stops improving, but only after MIN_EPOCHS
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,
            start_from_epoch=MIN_EPOCHS
        ),
        # save the model whenever val_loss improves
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # save training history so evaluate.py can plot the curves later
    import numpy as np
    np.save(HISTORY_PATH, history.history)

    print("\n" + "=" * 50)
    print(f"Training complete! Model saved to: {MODEL_PATH}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print("=" * 50)

    return model, history


if __name__ == "__main__":
    train_model()