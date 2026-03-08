import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
from io import BytesIO
import cv2
import joblib
import requests

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Plant Guardian & Crop Advisor", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .prediction-card { padding: 12px; border-radius: 10px; margin-top: 10px; font-size: 16px; text-align: center; background-color: #f0f2f6; color: #31333F; }
    h1, h3, p { text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_all_models():
    # Disease Model (CNN)
    disease_model = load_model("plant_disease_model_final4.h5")
    # Crop Model (Random Forest) - Update path as needed
    try:
        crop_model = joblib.load("rf_crop_recommendation.joblib")
    except:
        crop_model = None # Fallback if file isn't found locally
    return disease_model, crop_model

disease_model, crop_model = load_all_models()

# --- DATA MAPS ---
# (Keeping your class_names and maps from the original code)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Bitter Gourd__Downy_mildew', 'Bitter Gourd__Fusarium_wilt', 'Bitter Gourd__Fresh_leaf', 'Bitter Gourd__Mosaic_virus', 'Blueberry___healthy', 'Bottle gourd__Anthracnose', 'Bottle gourd__Downey_mildew', 'Bottle gourd__Fresh_leaf', 'Cauliflower__Black_Rot', 'Cauliflower__Downy_mildew', 'Cauliflower__Fresh_leaf', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Cucumber__Anthracnose_lesions', 'Cucumber__Downy_mildew', 'Cucumber__Fresh_leaf', 'Eggplant_Cercopora_leaf_spot', 'Eggplant_begomovirus', 'Eggplant_fresh_leaf', 'Eggplant_verticillium_wilt', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Guava_Healthy', 'Guava_Phytopthora', 'Guava_Red_rust', 'Guava_Scab', 'Guava_Styler_and_Root', 'Orange___Haunglongbing_(Citrus_greening)', 'Paddy_Bacterial_leaf_blight', 'Paddy_Brown_spot', 'Paddy_Leaf_smut', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Sugarcane_Healthy', 'Sugarcane_Mosaic', 'Sugarcane_RedRot', 'Sugarcane_Rust', 'Sugarcane_Yellow', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Wheat_Healthy', 'Wheat_leaf_leaf_stripe_rust', 'Wheatleaf_septoria']

fertilizer_map = { 'Apple___Apple_scab': 'Use copper-based fungicides', 'Apple___Black_rot': 'Apply sulfur sprays or captan', 'Apple___Cedar_apple_rust': 'Use myclobutanil or mancozeb', 'Apple___healthy': 'No fertilizer needed', 'Bitter Gourd__Downy_mildew': 'Use mancozeb or copper oxychloride spray', 'Bitter Gourd__Fusarium_wilt': 'Apply Trichoderma and maintain soil pH', 'Bitter Gourd__Mosaic_virus': 'Remove infected plants; control whiteflies', 'Cauliflower__Black_Rot': 'Apply copper-based bactericide', 'Corn___Common_rust': 'Apply fungicides like propiconazole', 'Paddy_Bacterial_leaf_blight': 'Apply bleaching powder; spray streptocycline', 'Tomato___Late_blight': 'Spray with mancozeb or copper-based fungicide' } # Shortened for display

label_mapping = {0: "rice", 1: "wheat", 2: "maize", 3: "chickpea", 4: "kidneybeans", 5: "pigeonpeas", 6: "mothbeans", 7: "mungbean", 8: "blackgram", 9: "lentil", 10: "pomegranate", 11: "banana", 12: "mango", 13: "grapes", 14: "watermelon", 15: "muskmelon", 16: "apple", 17: "orange", 18: "papaya", 19: "coconut", 20: "cotton", 21: "jute", 22: "coffee"}

# (Include your full crop_info and fertilizer_advice dictionaries here)

# --- HELPER FUNCTIONS ---
def get_weather(city_name):
    API_KEY = "8c3a497f31607fe66be1f23c65538904"
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city_name, "appid": API_KEY, "units": "metric"}
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            return data["main"]["temp"], data["main"]["humidity"], data.get("rain", {}).get("1h", 0), None
        return None, None, None, response.json().get("message", "Error")
    except: return None, None, None, "Connection Error"

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    original_img = np.array(original_img)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)

# --- SIDEBAR ---
st.sidebar.title("🌿 Agri-Smart Hub")
st.sidebar.info("Switch between tabs to detect diseases or get crop recommendations based on your soil and local weather.")

# --- MAIN LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🔍 Disease Detection", "🌾 Crop Recommendation", "📘 Info"])

# TAB 1: DETECTION
with tab1:
    st.markdown("## 🍃 Plant Disease Detection")
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"], key="det_upload")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Leaf", width=300)
        
        # Preprocessing
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Analyze Health"):
            prediction = disease_model.predict(img_array)
            idx = np.argmax(prediction)
            res = class_names[idx].replace('___', ' - ')
            st.success(f"**Result:** {res} ({np.max(prediction)*100:.2f}%)")
            
            # Advice
            if class_names[idx] in fertilizer_map:
                st.warning(f"💊 **Treatment:** {fertilizer_map[class_names[idx]]}")
            
            # Grad-CAM
            try:
                heatmap = get_gradcam_heatmap(disease_model, img_array, "Conv_1")
                overlay = overlay_gradcam(img_resized, heatmap)
                st.image(overlay, caption="AI Focus Area")
            except: st.info("Grad-CAM visualization skipped for this model.")

# TAB 2: CROP RECOMMENDATION
with tab2:
    st.markdown("## 🚜 Smart Crop Recommendation")
    
    # Weather Integration
    col1, col2 = st.columns([2, 1])
    with col2:
        st.write("### 🌦️ Local Weather")
        city = st.text_input("Enter City", "Hyderabad")
        if st.button("Fetch Live Weather"):
            t, h, r, err = get_weather(city)
            if not err:
                st.session_state.temp, st.session_state.hum, st.session_state.rain = t, h, r
                st.success(f"Weather updated for {city}!")
            else: st.error(err)

    with col1:
        st.write("### 🧪 Soil & Environment")
        # Session state defaults
        temp = st.number_input("Temp (°C)", 0.0, 50.0, float(st.session_state.get('temp', 25.0)))
        hum = st.number_input("Humidity (%)", 0.0, 100.0, float(st.session_state.get('hum', 70.0)))
        rain = st.number_input("Rainfall (mm)", 0.0, 500.0, float(st.session_state.get('rain', 100.0)))
        
        n = st.number_input("Nitrogen (N)", 0, 200, 50)
        p = st.number_input("Phosphorus (P)", 0, 200, 50)
        k = st.number_input("Potassium (K)", 0, 200, 50)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

    if st.button("Recommend Best Crop"):
        if crop_model:
            features = np.array([[n, p, k, temp, hum, ph, rain]])
            pred_idx = crop_model.predict(features)[0]
            crop = label_mapping[int(pred_idx)]
            st.balloons()
            st.success(f"🌱 Recommended Crop: **{crop.capitalize()}**")
            
            # Display details from your dictionaries
            # (Logic for crop_info and fertilizer_advice here)
        else:
            st.error("Crop model file not found. Please check 'rf_crop_recommendation.joblib' path.")

# TAB 3: INFO
with tab3:
    st.markdown("## 📘 About DeepCropCare & Advisor")
    st.write("Combined platform for Precision Agriculture...")
