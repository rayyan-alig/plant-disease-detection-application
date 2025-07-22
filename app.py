import tensorflow as tf
from tensorflow.keras import mixed_precision
import streamlit as st
import numpy as np
from PIL import Image

mixed_precision.set_global_policy('mixed_float16')

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_densenet.keras")

model = load_model()

class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

plant_names = sorted(set(name.split("___")[0].replace("_(including_sour)", "").replace(",", "") for name in class_names))

with st.sidebar:
    st.title("🌱 Supported Plants")
    for plant in plant_names:
        st.markdown(f"- {plant}")
    st.markdown("---")
    st.warning("⚠️ This app is for educational and research purposes only, dont use it for agricultural decisions.")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0
    return np.expand_dims(img_array.astype(np.float32), axis=0)

st.title("🌿 Plant Disease Classifier")

uploaded_file = st.file_uploader("📤 Upload a plant leaf image with plain background", type=['jpg', 'png', 'jpeg', 'jfif'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Predicting..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        pred_index = np.argmax(prediction)
        confidence = prediction[0][pred_index]

    st.success(f"🧪 **Prediction:** `{class_names[pred_index]}`")
    st.info(f"🔬 **Confidence:** `{confidence:.2%}`")
