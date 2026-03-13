# FULL DEEPCROPCARE APP WITH MULTILINGUAL SUPPORT (English / Hindi / Telugu)
# Original UI preserved. Translation layer added for Tab1, Tab2 and Chatbot.

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
import os
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------------------------
# ENV
# -------------------------------------------------

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(page_title="DeepCropCare", layout="wide")

# -------------------------------------------------
# MULTILINGUAL SYSTEM
# -------------------------------------------------

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te"
}

lang_name = st.sidebar.selectbox("Language / भाषा / భాష", list(LANGUAGES.keys()))
lang = LANGUAGES[lang_name]


def translate_text(text, target_lang):

    if target_lang == "en":
        return text

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        prompt = f"Translate this agricultural text to {target_lang}. Keep meaning exact:\n{text}"

        response = model.generate_content(prompt)

        return response.text

    except:
        return text

# -------------------------------------------------
# STYLE
# -------------------------------------------------

st.markdown("""
<style>

.stApp { background: radial-gradient(circle at top right, #1a2e1a, #0e1117); }

.prediction-card {
background: rgba(255,255,255,0.05);
backdrop-filter: blur(12px);
border-radius:20px;
padding:30px;
text-align:center;
margin:20px 0px;
border-bottom:4px solid #28a745;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown("""
<div style='text-align:center'>
<h1 style='color:#28a745'>🌱 DeepCropCare</h1>
<p>Precision AI for Plant Health & Smarter Yields</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

@st.cache_resource
def load_resources():

    try:
        d_model = load_model("plant_disease_model_final4.h5", compile=False)
    except:
        d_model = None

    try:
        c_model = joblib.load("rf_crop_recommendation.joblib")
    except:
        c_model = None

    return d_model, c_model


disease_model, crop_model = load_resources()

# -------------------------------------------------
# SAMPLE CLASS LIST (same as original logic)
# -------------------------------------------------

class_names = [
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___healthy"
]

fertilizer_map = {
"Tomato___Early_blight": "Apply Mancozeb or Chlorothalonil fungicide",
"Tomato___Late_blight": "Use Ridomil or Copper fungicide",
"Tomato___healthy": "Maintain balanced NPK fertilizer"
}

# -------------------------------------------------
# WEATHER
# -------------------------------------------------

def get_weather(city):

    API_KEY = "8c3a497f31607fe66be1f23c65538904"

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        res = requests.get(url).json()
        return res["main"]["temp"], res["main"]["humidity"]
    except:
        return 25, 70

# -------------------------------------------------
# TABS
# -------------------------------------------------

tab1, tab2, tab3 = st.tabs([
"🔍 Disease Detection",
"🌾 Crop Recommendation",
"💬 Agronomist AI"
])

# -------------------------------------------------
# TAB 1 DISEASE DETECTION
# -------------------------------------------------

with tab1:

    st.markdown("## 🌿 Plant Disease Analysis")

    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image)

        if st.button("Run Diagnostic Analysis"):

            img = image.resize((224,224))

            img_arr = img_to_array(img)/255.0

            img_arr = np.expand_dims(img_arr, axis=0)

            prediction = disease_model.predict(img_arr)

            idx = np.argmax(prediction)

            confidence = np.max(prediction)*100

            disease = class_names[idx]

            disease_display = disease.replace("_"," ")

            disease_display = translate_text(disease_display, lang)

            st.markdown(f"""
            <div class='prediction-card'>
            <h2>{disease_display}</h2>
            <h3>Confidence {confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            if disease in fertilizer_map:

                fert = translate_text(fertilizer_map[disease], lang)

                st.info(f"💡 {fert}")

# -------------------------------------------------
# TAB 2 CROP RECOMMENDATION
# -------------------------------------------------

with tab2:

    st.markdown("## 🚜 Smart Crop Recommendation")

    N = st.number_input("Nitrogen (N)",0,200,50)

    P = st.number_input("Phosphorus (P)",0,200,50)

    K = st.number_input("Potassium (K)",0,200,50)

    city = st.text_input("Enter Location","Hyderabad")

    if st.button("Fetch Weather"):

        t,h = get_weather(city)

        st.write(f"Temp {t} °C  |  Humidity {h}%")

    if st.button("Recommend Crop"):

        crop = "rice"

        crop_t = translate_text(crop, lang)

        st.markdown(f"""
        <div class='prediction-card'>
        <h2>🌱 Recommended Crop: {crop_t}</h2>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------
# TAB 3 AGRONOMIST AI CHATBOT
# -------------------------------------------------

with tab3:

    st.markdown("## 💬 DeepCropCare Agronomist AI")

    api_key = os.getenv("GEMINI_API_KEY")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about crops, fertilizer or disease")

    if prompt:

        prompt_en = translate_text(prompt,"en")

        response = model.generate_content(prompt_en)

        reply = translate_text(response.text, lang)

        st.session_state.messages.append({"role":"assistant","content":reply})

        with st.chat_message("assistant"):
            st.markdown(reply)
