import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("Upload an image and the model will predict whether it is a Cat or Dog.")

st.write("App started")

# Load model only once
@st.cache_resource
def load_model():
    st.write("Loading model...")
    model = tf.keras.models.load_model("cat_dog_model.h5")
    st.write("Model loaded!")
    return model

model = load_model()

st.success("Everything is working!")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Predicting..."):
        prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("🐶 Dog")
        st.write(f"Confidence: {prediction[0][0] * 100:.2f}%")
    else:
        st.success("🐱 Cat")
        st.write(f"Confidence: {(1 - prediction[0][0]) * 100:.2f}%")
