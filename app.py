# Streamlit app for digit recognition
# lets users draw or upload a digit and get predictions from the trained CNN

import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from utils.helpers import (
    preprocess_image_for_prediction,
    get_prediction_confidence,
    validate_image_file
)


# page setup
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide"
)


MODEL_PATH = 'models/mnist_model.keras'


@st.cache_resource
def load_model():
    """loads the saved keras model, only runs once thanks to caching"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Run train.py first.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def draw_digit_tab(model):
    """the drawing canvas tab - user draws a digit and clicks predict"""
    st.subheader("Draw a Digit")

    # we use a counter to generate new keys for the canvas
    # changing the key forces streamlit to create a fresh canvas (thats how clear works)
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    col1, col2 = st.columns([3, 1])

    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=15,
            stroke_color="rgba(0, 0, 0, 1)",
            background_color="rgba(255, 255, 255, 1)",
            update_streamlit=True,
            height=300,
            width=300,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            return_image_data=True
        )

    with col2:
        st.write("### Instructions")
        st.write("1. Draw a digit (0-9)")
        st.write("2. Click **Predict**")
        st.write("3. Clear and try again")

        if st.button("Clear Canvas", type="secondary"):
            # bump the key so a new blank canvas gets created
            st.session_state.canvas_key += 1
            st.session_state.pop("prediction", None)
            st.rerun()

        if st.button("Predict from Drawing", type="primary"):
            if canvas_result.image_data is not None:
                # grab canvas image, convert to grayscale
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img = img.convert('L')

                # run it through the model
                img_processed = preprocess_image_for_prediction(img)
                prediction = model.predict(img_processed, verbose=0)
                predicted_digit, confidence, all_probs = get_prediction_confidence(prediction)

                st.session_state['prediction'] = {
                    'digit': predicted_digit,
                    'confidence': confidence,
                    'probabilities': all_probs
                }


def upload_image_tab(model):
    """tab where users can upload a digit image instead of drawing"""
    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose a digit image (PNG or JPG)",
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        try:
            validate_image_file(uploaded_file)

            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Uploaded Image", width=250)

            if st.button("Predict from Upload", type="primary"):
                img_processed = preprocess_image_for_prediction(image)
                prediction = model.predict(img_processed, verbose=0)
                predicted_digit, confidence, all_probs = get_prediction_confidence(prediction)

                st.session_state['prediction'] = {
                    'digit': predicted_digit,
                    'confidence': confidence,
                    'probabilities': all_probs
                }

        except ValueError as e:
            st.error(str(e))


def show_prediction_results():
    """shows the prediction output - digit, confidence, and probability chart"""
    if 'prediction' not in st.session_state:
        return

    pred = st.session_state['prediction']

    st.divider()
    st.subheader("🎯 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Digit", f"{pred['digit']}")

    with col2:
        st.metric("Confidence", f"{pred['confidence']:.2f}%")

    with col3:
        # green if confident, orange if kinda sure, red if not really
        color = "green" if pred['confidence'] > 80 else "orange" if pred['confidence'] > 50 else "red"
        st.markdown(f"**Confidence Level:** :{color}[{pred['confidence']:.1f}%]")

    st.write("### Probability Distribution")

    probs = pred['probabilities']
    digits = list(probs.keys())
    values = list(probs.values())

    chart_data = {f"Digit {d}": v for d, v in zip(digits, values)}
    st.bar_chart(chart_data, horizontal=False)

    # expandable section showing exact percentages
    with st.expander("See all probabilities"):
        for digit, prob in probs.items():
            bar = "█" * int(prob / 2)
            st.write(f"**{digit}**: {prob:6.2f}% {bar}")


def main():
    # sidebar with project info
    st.sidebar.title("🔢 MNIST CNN")
    st.sidebar.markdown("---")

    model = load_model()

    if model is None:
        st.warning("⚠️ No trained model found. Please run train.py first.")
        st.code("python train.py", language="bash")
        return

    st.title("MNIST Digit Recognition")
    st.markdown("Draw or upload a handwritten digit (0-9) for prediction using a CNN model.")

    tab1, tab2 = st.tabs(["✏️ Draw Digit", "📁 Upload Image"])

    with tab1:
        draw_digit_tab(model)

    with tab2:
        upload_image_tab(model)

    show_prediction_results()

    # some info about the model in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    st.sidebar.info(
        """
        **CNN Architecture:**
        - 3 Convolutional layers
        - MaxPooling after each block
        - Dense layer with Dropout
        - Softmax output (10 classes)
        """
    )


if __name__ == "__main__":
    main()