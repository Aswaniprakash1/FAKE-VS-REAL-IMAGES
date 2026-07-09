import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 🔹 Load model (update path if needed)
MODEL_PATH = "C:/Users/admin/Downloads/image.h5" # place your trained .h5 model here
model = tf.keras.models.load_model(MODEL_PATH)

# 🔹 Class labels (edit based on your dataset)
classes = ['REAL', 'FAKE']

# 🔹 App title
st.title("🧠 Real vs Fake Image Classifier")
st.write("Upload an image to check whether it is REAL or FAKE")

# 🔹 File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img) / 255.0

    # Handle grayscale images
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = np.expand_dims(img, axis=0)

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(img)
        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Show result
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

# 🔹 Footer
st.write("---")
st.write("Built using Streamlit + CNN Model (.h5)")