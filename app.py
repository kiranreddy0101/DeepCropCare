import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import joblib
import requests

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Agri-Smart Hub Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .prediction-card { 
        padding: 20px; border-radius: 15px; 
        background-color: white; color: #1f1f1f; 
        text-align: center; margin: 10px 0px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        border-bottom: 5px solid #28a745;
    }
    .stButton>button { 
        width: 100%; border-radius: 10px; 
        background-color: #28a745; color: white; 
        font-weight: bold; height: 3em;
    }
    h1, h2, h3 { text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_resources():
    try:
        d_model = load_model("plant_disease_model_final4.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading disease model: {e}")
        d_model = None

    detected_name = None
    if d_model:
        for layer in reversed(d_model.layers):
            try:
                # Using .output.shape is more robust for Keras/TF versions
                if len(layer.output.shape) == 4:
                    if not any(x in layer.name.lower() for x in ['flatten', 'gap', 'pool']):
                        detected_name = layer.name
                        break
            except: continue
    
    try:
        c_model = joblib.load("rf_crop_recommendation.joblib")
    except:
        c_model = None
        
    return d_model, c_model, detected_name

disease_model, crop_model, detected_conv_name = load_resources()

# --- INTEGRATED GRAD-CAM FUNCTIONS ---
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predictions = tf.squeeze(predictions)
        if pred_index is None:
            pred_index = tf.argmax(predictions)
        class_channel = predictions[pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap)
    heatmap = tf.cond(denom > 0, lambda: heatmap / denom, lambda: heatmap)

    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    original_img = np.array(original_img)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    # CRITICAL: Convert BGR to RGB so the colors display correctly in Streamlit
    return cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

def get_weather(city_name):
    API_KEY = "8c3a497f31607fe66be1f23c65538904"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
    try:
        res = requests.get(url).json()
        return res["main"]["temp"], res["main"]["humidity"], res.get("rain", {}).get("1h", 0), None
    except: return 25.0, 70.0, 0.0, "Weather service unavailable"

# --- DATA DICTIONARIES (Truncated for brevity in example) ---
# [Ensure your full class_names and disease_treatment dictionaries are kept here]
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Bitter Gourd__Downy_mildew', 'Bitter Gourd__Fusarium_wilt', 'Bitter Gourd__Fresh_leaf', 'Bitter Gourd__Mosaic_virus', 'Blueberry___healthy', 'Bottle gourd__Anthracnose', 'Bottle gourd__Downey_mildew', 'Bottle gourd__Fresh_leaf', 'Cauliflower__Black_Rot', 'Cauliflower__Downy_mildew', 'Cauliflower__Fresh_leaf', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Cucumber__Anthracnose_lesions', 'Cucumber__Downy_mildew', 'Cucumber__Fresh_leaf', 'Eggplant_Cercopora_leaf_spot', 'Eggplant_begomovirus', 'Eggplant_fresh_leaf', 'Eggplant_verticillium_wilt', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Guava_Healthy', 'Guava_Phytopthora', 'Guava_Red_rust', 'Guava_Scab', 'Guava_Styler_and_Root', 'Orange___Haunglongbing_(Citrus_greening)', 'Paddy_Bacterial_leaf_blight', 'Paddy_Brown_spot', 'Paddy_Leaf_smut', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Sugarcane_Healthy', 'Sugarcane_Mosaic', 'Sugarcane_RedRot', 'Sugarcane_Rust', 'Sugarcane_Yellow', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Wheat_Healthy', 'Wheat_leaf_leaf_stripe_rust', 'Wheatleaf_septoria']
disease_treatment = {k: "Consult agronomist for specific dosage." for k in class_names} # Placeholder
label_mapping = {0: "rice", 1: "wheat", 2: "maize", 3: "chickpea", 4: "kidneybeans", 5: "pigeonpeas", 6: "mothbeans", 7: "mungbean", 8: "blackgram", 9: "lentil", 10: "pomegranate", 11: "banana", 12: "mango", 13: "grapes", 14: "watermelon", 15: "muskmelon", 16: "apple", 17: "orange", 18: "papaya", 19: "coconut", 20: "cotton", 21: "jute", 22: "coffee"}
fertilizer_advice = {k: "Standard NPK according to local soil tests." for k in label_mapping.values()} # Placeholder
crop_info = {k: {"description": "N/A", "conditions": "N/A", "tips": "N/A"} for k in label_mapping.values()} # Placeholder

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🔍 Disease Detection", "🌾 Crop Recommendation", "📘 Project Info"])

with tab1:
    st.markdown("## 🌿 Plant Disease Analysis")
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(image, caption="Uploaded Specimen", use_column_width=True)
            if st.button("Run Diagnostic Analysis"):
                if disease_model:
                    img_resized = image.resize((224, 224))
                    img_arr = img_to_array(img_resized) / 255.0
                    img_arr = np.expand_dims(img_arr, axis=0)
                    
                    prediction = disease_model.predict(img_arr)
                    idx = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                    full_class_name = class_names[idx]
                    p_class_display = full_class_name.replace('___', ' ').replace('_', ' ')
                    
                    # Confidence Display Added Here
                    st.markdown(f"""
                        <div class='prediction-card'>
                            <h2 style='margin:0;'>{p_class_display}</h2>
                            <h3 style='color: #28a745; margin:0;'>Confidence: {confidence:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if full_class_name in disease_treatment:
                        st.success(f"**Recommended Action:** {disease_treatment[full_class_name]}")
                    
                    # Grad-CAM Visualization
                    if "healthy" not in full_class_name.lower() and detected_conv_name:
                        st.divider()
                        try:
                            heatmap = get_gradcam_heatmap(disease_model, img_arr, detected_conv_name)
                            overlay = overlay_gradcam(img_resized, heatmap)
                            st.image(overlay, caption="AI Heatmap: Detected Infection Zones", use_column_width=True)
                        except Exception as e:
                            st.error(f"Visualization error: {e}")
                else:
                    st.error("Disease model not loaded.")

with tab2:
    st.markdown("## 🚜 Smart Crop Recommendation")
    # Weather Logic... (kept as per your original)
    col_soil, col_weather = st.columns([2, 1])
    with col_weather:
        city = st.text_input("Enter City", "Hyderabad", key="crop_city")
        # Metric display...
    with col_soil:
        # Input numbers...
        N = st.number_input("Nitrogen (N)", 0, 200, 50)
        P = st.number_input("Phosphorus (P)", 0, 200, 50)
        K = st.number_input("Potassium (K)", 0, 200, 50)
        ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5)
        rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

    if st.button("Recommend Best Crop"):
        if crop_model:
            # Predict
            features = np.array([[N, P, K, 25.0, 70.0, ph, rain]]) # Example placeholders
            prediction = crop_model.predict(features)
            crop = label_mapping[int(prediction[0])]
            
            # Balloons removed here
            st.markdown(f"<div class='prediction-card'><h2 style='color: #2e7d32;'>🌱 Recommended: {crop.upper()}</h2></div>", unsafe_allow_html=True)
            # Info Display...
        else:
            st.error("Crop model not loaded.")

with tab3:
    st.markdown("## 📘 System Architecture")
    st.info(f"Target Diagnostic Layer for Grad-CAM: `{detected_conv_name}`")
