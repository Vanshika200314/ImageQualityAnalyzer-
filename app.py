# The line above is a special command that writes the content of this cell to a file named 'app.py'

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# --- App Title and Description ---
st.set_page_config(page_title="Image Quality Analyzer", layout="centered")
st.title("🖼️ AI-Powered Image Quality Analyzer")
st.write(
    "Upload an image and our AI model will classify it as either "
    "'Good Quality' or 'Bad Quality'. The model is trained to detect issues "
    "like blur, darkness, and low contrast."
)

# --- Function to Load and Cache the Model ---
# @st.cache_resource is a decorator that tells Streamlit to load the model only once,
# which makes the app much faster.
@st.cache_resource
def load_our_model():
    # We need to specify the full path to the model in Google Drive
    model_path = '/content/drive/MyDrive/ImageQualityProject/image_quality_analyzer.h5'
    model = tf.keras.models.load_model(model_path)
    return model

# Load the model
model = load_our_model()

# --- Image Upload and Prediction ---
uploaded_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # --- Display the Uploaded Image and a Separator ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Your Uploaded Image', use_column_width=True)

    # --- Preprocess the Image for the Model ---
    # Convert the image to a NumPy array
    img_array = np.array(image)
    # Resize to the model's expected input size (128x128)
    img_array = cv2.resize(img_array, (128, 128))
    # Normalize the pixel values to be between 0 and 1
    img_array = img_array / 255.0
    # Add a batch dimension, as the model expects a batch of images
    img_array = np.expand_dims(img_array, axis=0)

    # --- Make a Prediction ---
    with st.spinner('🧠 Analyzing image...'):
        prediction = model.predict(img_array)

    # The prediction is a probability. We check if it's > 0.5 to classify as "Bad".
    is_bad_quality = prediction[0][0] > 0.5

    # --- Display the Result ---
    with col2:
        st.subheader("Analysis Result:")
        if is_bad_quality:
            st.error("Prediction: Bad Quality Image")
            st.write(
                "Our model has detected potential issues such as blurriness, "
                "darkness, or low contrast in this image."
            )
        else:
            st.success("Prediction: Good Quality Image")
            st.write(
                "Our model believes this image is sharp, well-lit, and of good quality."
            )

        # Display the raw prediction score for transparency
        st.write(f"**Model's Confidence Score:** {prediction[0][0]:.4f}")
        st.info(
            "A score closer to 1.0 indicates a higher probability of being 'Bad Quality'.\n"
            "A score closer to 0.0 indicates a higher probability of being 'Good Quality'."
        )

else:
    st.info("Please upload an image file to see the analysis.")
