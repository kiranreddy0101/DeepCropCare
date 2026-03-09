import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import joblib
import requests
import time

# --- CONFIG & STYLING ---
st.set_page_config(page_title="DeepCropCare", layout="wide")

# --- TRANSLATION ENGINE ---
languages = {
    "English": {
        "title": "🌱 DeepCropCare",
        "subtitle": "Precision AI for Plant Health & Smarter Yields",
        "tab1": "🔍 Disease Detection",
        "tab2": "🌾 Crop Recommendation",
        "tab3": "📘 Project Info",
        "analysis_header": "🌿 Plant Disease Analysis",
        "upload_label": "Upload leaf image",
        "run_btn": "Run Diagnostic Analysis",
        "identifying": "🧠 Identifying pathogens...",
        "complete": "Analysis Complete!",
        "confidence": "Confidence",
        "action": "Recommended Action",
        "heatmap": "🎯 AI Heatmap: Detected Infection Zones",
        "orig_scan": "Original Scan",
        "hotspots": "Infection Hotspots",
        "smart_crop": "🚜 Smart Crop Recommendation",
        "soil_params": "🧪 Soil Parameters",
        "local_weather": "🌦️ Local Weather",
        "fetch_weather": "Fetch Live Weather",
        "rec_btn": "Recommend Best Crop",
        "rec_label": "Recommended",
        "desc_label": "📖 Description",
        "cond_label": "🔍 Optimal Conditions",
        "fert_label": "🧪 Fertilizer & Care Advice",
        "pro_tip": "Pro-Tip"
    },
    "Hindi": {
        "title": "🌱 डीप क्रॉप केयर",
        "subtitle": "पौधों के स्वास्थ्य और स्मार्ट उपज के लिए सटीक एआई",
        "tab1": "🔍 रोग की पहचान",
        "tab2": "🌾 फसल अनुशंसा",
        "tab3": "📘 प्रोजेक्ट जानकारी",
        "analysis_header": "🌿 पादप रोग विश्लेषण",
        "upload_label": "पत्ती की छवि अपलोड करें",
        "run_btn": "नैदानिक विश्लेषण चलाएं",
        "identifying": "🧠 रोगजनकों की पहचान...",
        "complete": "विश्लेषण पूर्ण!",
        "confidence": "आत्मविश्वास",
        "action": "अनुशंसित कार्रवाई",
        "heatmap": "🎯 एआई हीटमैप: संक्रमण क्षेत्र",
        "orig_scan": "मूल स्कैन",
        "hotspots": "संक्रमण हॉटस्पॉट",
        "smart_crop": "🚜 स्मार्ट फसल अनुशंसा",
        "soil_params": "🧪 मिट्टी के पैरामीटर",
        "local_weather": "🌦️ स्थानीय मौसम",
        "fetch_weather": "लाइव मौसम प्राप्त करें",
        "rec_btn": "सर्वोत्तम फसल की सिफारिश करें",
        "rec_label": "अनुशंसित",
        "desc_label": "📖 विवरण",
        "cond_label": "🔍 अनुकूल परिस्थितियां",
        "fert_label": "🧪 उर्वरक और देखभाल सलाह",
        "pro_tip": "प्रो-टिप"
    },
    "Telugu": {
        "title": "🌱 డీప్ క్రాప్ కేర్",
        "subtitle": "మొక్కల ఆరోగ్యం & మెరుగైన దిగుబడి కోసం ఖచ్చితమైన AI",
        "tab1": "🔍 వ్యాధి గుర్తింపు",
        "tab2": "🌾 పంట సిఫార్సు",
        "tab3": "📘 ప్రాజెక్ట్ సమాచారం",
        "analysis_header": "🌿 మొక్కల వ్యాధి విలేషణ",
        "upload_label": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "run_btn": "విశ్లేషణను ప్రారంభించండి",
        "identifying": "🧠 వ్యాధిని గుర్తిస్తోంది...",
        "complete": "విశ్లేషణ పూర్తయింది!",
        "confidence": "ఖచ్చితత్వం",
        "action": "సిఫార్సు చేయబడిన చర్య",
        "heatmap": "🎯 AI హీట్‌మ్యాప్: వ్యాధి సోకిన ప్రాంతాలు",
        "orig_scan": "అసలు స్కాన్",
        "hotspots": "వ్యాధి హాట్‌స్పాట్‌లు",
        "smart_crop": "🚜 స్మార్ట్ పంట సిఫార్సు",
        "soil_params": "🧪 నేల పారామితులు",
        "local_weather": "🌦️ స్థానిక వాతావరణం",
        "fetch_weather": "వాతావరణ సమాచారాన్ని పొందండి",
        "rec_btn": "ఉత్తమ పంటను సిఫార్సు చేయండి",
        "rec_label": "సిఫార్సు చేయబడినది",
        "desc_label": "📖 వివరణ",
        "cond_label": "🔍 అనుకూల పరిస్థితులు",
        "fert_label": "🧪 ఎరువులు & సంరక్షణ సలహా",
        "pro_tip": "చిట్కా"
    }
}

# Language Selector
st.sidebar.title("🌐 Language Selection")
lang_choice = st.sidebar.selectbox("Choose Language", options=list(languages.keys()))
T = languages[lang_choice]

st.markdown("""
    <style>
    .stAppDeployButton {display:none;}
    .block-container { padding-top: 1.5rem !important; margin-top: -4rem !important; max-width: 95%; }
    .top-header { text-align: center; padding-bottom: 1rem; }
    .stApp { background: radial-gradient(circle at top right, #1a2e1a, #0e1117); color: white; }
    div.stButton > button {
        width: 100% !important; white-space: nowrap !important;
        font-weight: bold !important; background-color: #28a745 !important;
        color: white !important; border-radius: 10px !important;
        padding: 0.5rem 1rem !important; font-size: 1rem !important;
    }
    .prediction-card { 
        background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; 
        padding: 30px; text-align: center; margin: 20px 0px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8); border-bottom: 4px solid #28a745;
    }
    .prediction-card h2 { color: #28a745 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="top-header">
        <h1 style="font-size: 3.5rem; color: #28a745; margin-bottom: 0;">{T['title']}</h1>
        <p style="font-size: 1.1rem; color: #a3a3a3;">{T['subtitle']}</p>
    </div>
""", unsafe_allow_html=True)

# --- GRAD-CAM FUNCTIONS ---
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
    return cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    try: d_model = load_model("plant_disease_model_final4.h5", compile=False)
    except: d_model = None

    detected_name = None
    if d_model:
        for layer in reversed(d_model.layers):
            if len(layer.output.shape) == 4:
                if not any(x in layer.name.lower() for x in ['flatten', 'gap', 'pool']):
                    detected_name = layer.name
                    break
    
    try: c_model = joblib.load("rf_crop_recommendation.joblib")
    except: c_model = None
    return d_model, c_model, detected_name

disease_model, crop_model, detected_conv_name = load_resources()

def get_weather(city_name):
    API_KEY = "8c3a497f31607fe66be1f23c65538904"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
    try:
        res = requests.get(url).json()
        return res["main"]["temp"], res["main"]["humidity"], None
    except: return 25.0, 70.0, "Service unavailable"

# --- DATA DICTIONARIES ---
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Bitter Gourd__Downy_mildew', 'Bitter Gourd__Fusarium_wilt',
    'Bitter Gourd__Fresh_leaf', 'Bitter Gourd__Mosaic_virus',
    'Blueberry___healthy', 'Bottle gourd__Anthracnose', 'Bottle gourd__Downey_mildew',
    'Bottle gourd__Fresh_leaf', 'Cauliflower__Black_Rot', 'Cauliflower__Downy_mildew',
    'Cauliflower__Fresh_leaf', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Cucumber__Anthracnose_lesions',
    'Cucumber__Downy_mildew', 'Cucumber__Fresh_leaf',
    'Eggplant_Cercopora_leaf_spot', 'Eggplant_begomovirus', 'Eggplant_fresh_leaf',
    'Eggplant_verticillium_wilt', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Guava_Healthy',
    'Guava_Phytopthora', 'Guava_Red_rust', 'Guava_Scab', 'Guava_Styler_and_Root',
    'Orange___Haunglongbing_(Citrus_greening)', 'Paddy_Bacterial_leaf_blight',
    'Paddy_Brown_spot', 'Paddy_Leaf_smut', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Sugarcane_Healthy', 'Sugarcane_Mosaic', 'Sugarcane_RedRot',
    'Sugarcane_Rust', 'Sugarcane_Yellow',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy', 'Wheat_Healthy', 'Wheat_leaf_leaf_stripe_rust', 'Wheatleaf_septoria'
]

fertilizer_map = {
    'Apple___Apple_scab': 'Use copper-based fungicides (Liquid Copper)',
    'Apple___Black_rot': 'Apply sulfur sprays or captan; prune cankers',
    'Apple___Cedar_apple_rust': 'Use myclobutanil or mancozeb during spring',
    'Apple___healthy': 'Maintain soil organic matter with compost',
    'Bitter Gourd__Downy_mildew': 'Apply Mancozeb or Copper Oxychloride spray',
    'Bitter Gourd__Fusarium_wilt': 'Soil drenching with Carbendazim (0.1%)',
    'Bitter Gourd__Mosaic_virus': 'Control aphids with Neem Oil; remove infected vines',
    'Bottle gourd__Anthracnose': 'Apply Chlorothalonil or Mancozeb',
    'Cauliflower__Black_Rot': 'Seed treatment with Streptocycline; Copper Oxychloride',
    'Corn___Common_rust': 'Foliar spray of Mancozeb (0.2%)',
    'Grape___Black_rot': 'Spray Captan or Mancozeb at bloom stage',
    'Paddy_Bacterial_leaf_blight': 'Spray Streptocycline + Copper Oxychloride',
    'Potato___Late_blight': 'Metalaxyl-M + Mancozeb (Ridomil)',
    'Tomato___Early_blight': 'Apply Chlorothalonil or Azoxystrobin',
    'Wheat_leaf_leaf_stripe_rust': 'Apply Propiconazole (Tilt) 25 EC'
}

label_mapping = {
    0: "rice", 1: "wheat", 2: "maize", 3: "chickpea", 4: "kidneybeans",
    5: "pigeonpeas", 6: "mothbeans", 7: "mungbean", 8: "blackgram", 9: "lentil",
    10: "pomegranate", 11: "banana", 12: "mango", 13: "grapes", 14: "watermelon",
    15: "muskmelon", 16: "apple", 17: "orange", 18: "papaya", 19: "coconut",
    20: "cotton", 21: "jute", 22: "coffee"
}

fertilizer_advice = {
    "rice": "Apply Urea, DAP, and MOP in split doses; ensure timely irrigation.",
    "wheat": "Use nitrogen-rich fertilizer (urea) and phosphorus (DAP) before sowing.",
    "maize": "Balanced NPK application with nitrogen split at 30 & 45 days.",
    "coffee": "Use organic compost and potassium sulfate; apply nitrogen after pruning."
}

crop_info = {
    "rice": {
        "description": "Rice is a staple food crop grown in warm, humid climates.",
        "conditions": "Grows best in clayey loam soil, temp 20–35°C.",
        "tips": "Maintain standing water in fields."
    },
    "wheat": {
        "description": "Wheat is a major cereal crop grown in temperate regions.",
        "conditions": "Requires cool weather during early growth.",
        "tips": "Use certified seeds; apply nitrogen in split doses."
    }
}

# --- TABS ---
tab1, tab2, tab3 = st.tabs([T["tab1"], T["tab2"], T["tab3"]])

with tab1:
    st.markdown(f"## {T['analysis_header']}")
    uploaded_file = st.file_uploader(T["upload_label"], type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col_s1, col_img, col_s2 = st.columns([1, 0.8, 1])
        with col_img:
            st.image(image, caption=T["orig_scan"], use_container_width=True)
        
        _, center_col, _ = st.columns([1, 1, 1])
        with center_col:
            run_btn = st.button(T["run_btn"])
        
        if run_btn: 
            progress_bar = st.progress(0)
            for p in range(100):
                time.sleep(0.01)
                progress_bar.progress(p + 1)
    
            with st.spinner(T["identifying"]):
                if disease_model:
                    img_resized = image.resize((224, 224))
                    img_arr = img_to_array(img_resized) / 255.0
                    img_arr = np.expand_dims(img_arr, axis=0)
                    prediction = disease_model.predict(img_arr)
                    idx = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                    full_class_name = class_names[idx]
                    p_class_display = full_class_name.replace('___', ' ').replace('_', ' ')
                    
                    st.markdown(f"""
                        <div class='prediction-card'>
                            <h2>{p_class_display}</h2>
                            <h3>{T['confidence']}: {confidence:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if full_class_name in fertilizer_map:
                        st.info(f"**💡 {T['action']}:** {fertilizer_map[full_class_name]}")
                    
                    if "healthy" not in full_class_name.lower() and detected_conv_name:
                        st.markdown(f"<h3 style='text-align: center;'>{T['heatmap']}</h3>", unsafe_allow_html=True)
                        heatmap = get_gradcam_heatmap(disease_model, img_arr, detected_conv_name)
                        overlay = overlay_gradcam(img_resized, heatmap)
                        col_a, col_b = st.columns(2)
                        with col_a: st.image(img_resized, caption=T["orig_scan"])
                        with col_b: st.image(overlay, caption=T["hotspots"])

with tab2:
    st.markdown(f"## {T['smart_crop']}")
    if "weather_temp" not in st.session_state: st.session_state.weather_temp = 25.0
    if "weather_hum" not in st.session_state: st.session_state.weather_hum = 70.0

    col_soil, col_weather = st.columns([1.5, 1])
    with col_soil:
        st.write(f"### {T['soil_params']}")
        n1, p1, k1 = st.columns(3)
        N = n1.number_input("N", 0, 200, 50)
        P = p1.number_input("P", 0, 200, 50)
        K = k1.number_input("K", 0, 200, 50)
        ph = st.slider("pH", 0.0, 14.0, 6.5)
        rain = st.number_input("Rainfall (mm)", 0.0, 1000.0, 100.0)

    with col_weather:
        st.write(f"### {T['local_weather']}")
        city = st.text_input("City", "Kothur")
        if st.button(T["fetch_weather"]):
            t, h, _ = get_weather(city)
            st.session_state.weather_temp, st.session_state.weather_hum = float(t), float(h)
        st.write(f"Temp: {st.session_state.weather_temp}°C | Humidity: {st.session_state.weather_hum}%")

    if st.button(T["rec_btn"]):
        crop = label_mapping[crop_model.predict([[N,P,K,st.session_state.weather_temp, st.session_state.weather_hum, ph, rain]])[0]] if crop_model else "rice"
        st.markdown(f"<div class='prediction-card'><h2>🌱 {T['rec_label']}: {crop.upper()}</h2></div>", unsafe_allow_html=True)
        inf1, inf2 = st.columns(2)
        with inf1:
            st.markdown(f"### {T['desc_label']}")
            st.write(crop_info.get(crop, {}).get('description', 'Information not available.'))
        with inf2:
            st.markdown(f"### {T['fert_label']}")
            st.warning(fertilizer_advice.get(crop, "General fertilizer application recommended."))

with tab3:
    st.markdown("## 📘 DeepCropCare Project Info")
    st.write("This platform uses Deep Learning (CNN) and Random Forest models to assist farmers.")
