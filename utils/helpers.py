# helper functions used by the streamlit app
# mainly image preprocessing and formatting prediction output

import numpy as np
from PIL import Image


def preprocess_image_for_prediction(image):
    """
    takes whatever image the user gives us (from canvas or upload)
    and gets it into the format the model expects: 28x28 grayscale, normalized.

    also handles the background color thing - MNIST images have black backgrounds
    with white digits, but most people draw dark on light. so if the image is
    mostly bright, we invert it.
    """
    # handle numpy arrays (from canvas)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # make sure its grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # resize to 28x28 to match what the model was trained on
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # convert to float and normalize to 0-1
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0

    # invert colors if the background looks white
    # MNIST digits are white-on-black, most uploads are black-on-white
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array

    # reshape to match model input: (batch_size, height, width, channels)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


def get_prediction_confidence(prediction):
    """
    takes the raw softmax output and returns:
    - which digit it thinks it is
    - how confident it is (as a percentage)
    - probabilities for all 10 digits
    """
    probabilities = prediction[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit]) * 100

    all_probabilities = {
        str(i): float(probabilities[i]) * 100 for i in range(10)
    }

    return predicted_digit, confidence, all_probabilities


def validate_image_file(uploaded_file):
    """basic checks on uploaded files - right type and not too big"""
    allowed_types = ['image/png', 'image/jpeg', 'image/jpg']

    if uploaded_file.type not in allowed_types:
        raise ValueError(f"Invalid file type. Allowed: PNG, JPG, JPEG")

    # 5MB should be more than enough for a digit image
    if uploaded_file.size > 5 * 1024 * 1024:
        raise ValueError("File too large. Maximum size: 5MB")

    return True


def create_probability_chart(probabilities):
    """splits the probability dict into two lists for plotting"""
    digits = list(probabilities.keys())
    values = list(probabilities.values())
    return digits, values