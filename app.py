import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown

# --- Model Download ---
MODEL_PATH = "/tmp/waste_sorter_optimized.keras"

def download_model():
    file_id = "1iEX8JOJrt6CRRMDEzuYfPEcM5JxSSAA0"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... this may take a minute."):
        download_model()

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
    st.error("❌ Model download failed. Make sure the Google Drive file is shared as 'Anyone with the link'.")
    st.stop()

# --- Load Model ---
@st.cache_resource
def load_waste_model():
    return load_model(MODEL_PATH)

model = load_waste_model()

# --- Class Labels ---
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --- Disposal Tips ---
disposal_tips = {
    'cardboard': "📦 Flatten and place in the recycling bin. Remove any tape or staples.",
    'glass':     "🍶 Rinse and place in glass recycling. Do not mix with broken glass.",
    'metal':     "🥫 Rinse cans and place in recycling. Scrap metal can go to a metal recycler.",
    'paper':     "📄 Keep dry and place in paper recycling. Avoid greasy or food-soiled paper.",
    'plastic':   "🧴 Check the resin code. Bottles and containers are usually recyclable.",
    'trash':     "🗑️ This item belongs in general waste. Not recyclable.",
}

# --- UI ---
st.title("♻️ Waste Sorter")
st.write("Upload an image of waste and the model will classify it for you.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Results
    st.markdown("---")
    st.markdown(f"### 🔍 Result: `{predicted_class.upper()}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))
    st.info(disposal_tips[predicted_class])