# evaluates the trained model on the MNIST test set
# generates metrics, plots, confusion matrix, and classification report

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

from preprocessing import load_data, preprocess_data
from model import build_cnn_model


MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'mnist_model.keras')
RESULTS_DIR = 'results'


def load_trained_model():
    """loads the saved model file, throws an error if it doesnt exist yet"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
    return model


def plot_training_history():
    """plots accuracy and loss over epochs from the saved training history"""
    history_path = os.path.join(MODEL_DIR, 'history.npy')

    if not os.path.exists(history_path):
        print("Warning: History file not found, skipping training plots.")
        return

    history = np.load(history_path, allow_pickle=True).item()

    plt.figure(figsize=(10, 4))

    # accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='s')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_plots.png'), dpi=150)
    plt.close()
    print("Saved: training_plots.png")


def plot_confusion_matrix(y_true, y_pred):
    """creates a heatmap of the confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()

    classes = list(range(10))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # put the actual numbers in each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")


def evaluate_model():
    """
    full evaluation pipeline:
    load model → run on test set → compute all the metrics → save plots and report
    """
    print("=" * 50)
    print("MNIST Digit Recognition - Model Evaluation")
    print("=" * 50)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/4] Loading trained model...")
    model = load_trained_model()

    print("\n[2/4] Loading test data...")
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test
    )

    # run model on the test set
    print("\n[3/4] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # get predictions so we can compute precision, recall, etc
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nPrecision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")

    # detailed per-digit breakdown
    print("\n[4/4] Generating classification report...")
    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification Report:")
    print(report)

    # write everything to a text file
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("MNIST Digit Recognition - Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1-Score (weighted): {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(report)
    print(f"Saved: classification_report.txt")

    # make the plots
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_training_history()

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("=" * 50)

    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


if __name__ == "__main__":
    evaluate_model()