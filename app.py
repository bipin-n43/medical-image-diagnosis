import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Uncomment the following lines to load your pre-trained model
from tensorflow.keras.models import load_model
model = load_model('notebooks/cnn_model.h5')  # Replace with your model file

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .uploaded-image {
        display: block;
        margin: 20px auto;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .output-text {
        text-align: center;
        font-size: 20px;
        margin-top: 20px;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Application title
st.title("Alzheimer's Disease Detection")
st.markdown("### Upload an MRI Image to Detect Alzheimer's Disease")

# Upload an image
uploaded_file = st.file_uploader("Choose an MRI image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)  # Corrected line
    # Preprocess the image
    def preprocess_image(image):
        image = image.resize((128, 128))  # Resizing to model's input size
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        if len(image_array.shape) == 2:  # Convert grayscale to RGB if needed
            image_array = np.stack([image_array] * 3, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    labels = {0: "Mild Demented", 1: "Moderate Demented", 2: "Non Demented", 3: "Very Mild Demented"}
    result = labels[predicted_class]

    # For demonstration, display placeholder text
    # Display the result
    st.markdown(f"<div class='output-text'>Prediction: <b>{result}</b></div>", unsafe_allow_html=True)
else:
    st.info("Please upload an MRI image to proceed.")
