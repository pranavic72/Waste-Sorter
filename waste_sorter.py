import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image


import os
import gdown

MODEL_PATH = "waste_sorter_optimized.keras"  # or 'waste_sorter.h5', whatever your filename is

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    file_id = "1iEX8JOJrt6CRRMDEzuYfPEcM5JxSSAA0"  # Replace with your actual file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    print("Model file already exists locally.")




# --- Load Your Trained Model ---
@st.cache_resource
def load_waste_model():
    model = load_model(MODEL_PATH)  # make sure this file is in the same folder
    return model

model = load_waste_model()

# --- Define Class Labels (Adjust to Your Dataset Classes) ---
class_names = ['.ipynb_checkpoints','cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Example classes, modify as needed


# --- Streamlit UI ---
st.title("♻ Waste Sorter")
st.write("Upload an image of waste and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = img.resize((224, 224))  # match your model input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.markdown(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")

