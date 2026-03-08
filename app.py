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
    'Background_without_leaves': 'N/A',
    'Bitter Gourd__Downy_mildew': 'Apply Mancozeb or Copper Oxychloride spray',
    'Bitter Gourd__Fusarium_wilt': 'Soil drenching with Carbendazim (0.1%)',
    'Bitter Gourd__Fresh_leaf': 'Apply balanced NPK (10-10-10)',
    'Bitter Gourd__Mosaic_virus': 'Control aphids with Neem Oil; remove infected vines',
    'Blueberry___healthy': 'Apply acidic fertilizers (Ammonium Sulfate)',
    'Bottle gourd__Anthracnose': 'Apply Chlorothalonil or Mancozeb',
    'Bottle gourd__Downey_mildew': 'Use Metalaxyl or copper-based fungicides',
    'Bottle gourd__Fresh_leaf': 'Apply well-rotted farmyard manure',
    'Cauliflower__Black_Rot': 'Seed treatment with Streptocycline; Copper Oxychloride',
    'Cauliflower__Downy_mildew': 'Apply Ridomil Gold or Mancozeb',
    'Cauliflower__Fresh_leaf': 'Urea top-dressing for leafy growth',
    'Cherry___Powdery_mildew': 'Wettable sulfur or Myclobutanil sprays',
    'Cherry___healthy': 'Potassium-rich fertilizers during fruiting',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Apply Tilt (Propiconazole) fungicide',
    'Corn___Common_rust': 'Foliar spray of Mancozeb (0.2%)',
    'Corn___Northern_Leaf_Blight': 'Use resistant hybrids and apply Azoxystrobin',
    'Corn___healthy': 'Side-dress with Nitrogen (Urea) at knee height',
    'Cucumber__Anthracnose_lesions': 'Apply Chlorothalonil; avoid overhead irrigation',
    'Cucumber__Downy_mildew': 'Systemic fungicides like Metalaxyl',
    'Cucumber__Fresh_leaf': 'Apply liquid seaweed fertilizer',
    'Eggplant_Cercopora_leaf_spot': 'Mancozeb or Zineb sprays every 10 days',
    'Eggplant_begomovirus': 'Vector control (Whiteflies) using Imidacloprid',
    'Eggplant_fresh_leaf': 'NPK 15:15:15 application',
    'Eggplant_verticillium_wilt': 'Soil solarization and Trichoderma viride',
    'Grape___Black_rot': 'Spray Captan or Mancozeb at bloom stage',
    'Grape___Esca_(Black_Measles)': 'Pruning wound protection with fungicides',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Bordeaux mixture spray (1%)',
    'Grape___healthy': 'Apply Muriate of Potash (MOP) for sugar content',
    'Guava_Healthy': 'NPK 6:6:6 + Micronutrient spray (Zinc)',
    'Guava_Phytopthora': 'Improve drainage; apply Aliette (Fosetyl-Al)',
    'Guava_Red_rust': 'Copper oxychloride (0.3%) spray',
    'Guava_Scab': 'Foliar spray of Carbendazim',
    'Guava_Styler_and_Root': 'Zinc and Boron soil application',
    'Orange___Haunglongbing_(Citrus_greening)': 'Control Citrus Psyllid; apply Micronutrients',
    'Paddy_Bacterial_leaf_blight': 'Spray Streptocycline + Copper Oxychloride',
    'Paddy_Brown_spot': 'Apply Potash; spray Edifenphos or Mancozeb',
    'Paddy_Leaf_smut': 'Propiconazole (0.1%) spray',
    'Peach___Bacterial_spot': 'Copper-based sprays during dormancy',
    'Peach___healthy': 'Balanced NPK in early spring',
    'Pepper_bell___Bacterial_spot': 'Copper Hydroxide sprays; use disease-free seeds',
    'Pepper_bell___healthy': 'Apply 5-10-10 NPK',
    'Potato___Early_blight': 'Spray Mancozeb or Chlorothalonil',
    'Potato___Late_blight': 'Metalaxyl-M + Mancozeb (Ridomil)',
    'Potato___healthy': 'High Nitrogen during early growth',
    'Raspberry___healthy': 'Apply 10-10-10 NPK in spring',
    'Soybean___healthy': 'Phosphorus and Rhizobium inoculation',
    'Squash___Powdery_mildew': 'Neem Oil or Sulfur sprays',
    'Strawberry___Leaf_scorch': 'Avoid excess Nitrogen; apply Copper fungicides',
    'Strawberry___healthy': 'Phosphorus-rich fertilizer for berries',
    'Sugarcane_Healthy': 'Balanced NPK + Iron/Zinc if yellowing',
    'Sugarcane_Mosaic': 'Use virus-free setts; control aphids',
    'Sugarcane_RedRot': 'Treat setts with Carbendazim; improve drainage',
    'Sugarcane_Rust': 'Spray Pyraclostrobin or Mancozeb',
    'Sugarcane_Yellow': 'Apply Ferrous Sulfate (0.5%) foliar spray',
    'Tomato___Bacterial_spot': 'Copper spray + Streptocycline',
    'Tomato___Early_blight': 'Apply Chlorothalonil or Azoxystrobin',
    'Tomato___Late_blight': 'Spray Mancozeb (0.25%) or Ridomil',
    'Tomato___Leaf_Mold': 'Increase ventilation; apply Copper fungicide',
    'Tomato___Septoria_leaf_spot': 'Chlorothalonil or Mancozeb sprays',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Abamectin or Neem Oil spray',
    'Tomato___Target_Spot': 'Apply Pyraclostrobin or Boscalid',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Yellow sticky traps; control Whiteflies',
    'Tomato___Tomato_mosaic_virus': 'Sanitize tools; use resistant varieties',
    'Tomato___healthy': 'Regular NPK 10-10-10 + Calcium to prevent blossom end rot',
    'Wheat_Healthy': 'DAP + Urea top-dressing',
    'Wheat_leaf_leaf_stripe_rust': 'Apply Propiconazole (Tilt) 25 EC',
    'Wheatleaf_septoria': 'Spray Tebuconazole or Chlorothalonil'
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
    "wheat": "Use nitrogen-rich fertilizer (urea) and phosphorus (DAP) before sowing; top-dress nitrogen at tillering stage.",
    "maize": "Balanced NPK application with nitrogen split at 30 & 45 days; use zinc sulphate if deficient.",
    "chickpea": "Use Rhizobium inoculant before sowing and phosphorus-rich fertilizers; avoid excess nitrogen.",
    "kidneybeans": "Use balanced NPK with high phosphorus; apply well-rotted manure before planting.",
    "pigeonpeas": "Apply DAP or SSP at sowing; use organic compost to maintain soil health.",
    "mothbeans": "Use moderate NPK with emphasis on phosphorus; apply farmyard manure.",
    "mungbean": "Apply Rhizobium culture and balanced NPK; avoid excess nitrogen.",
    "blackgram": "Use Rhizobium and phosphate solubilizing bacteria; apply SSP at sowing.",
    "lentil": "Apply phosphorus and potassium; treat seeds with Rhizobium before sowing.",
    "pomegranate": "Apply high potassium (MOP) during fruit development; use compost annually.",
    "banana": "High potassium requirement; apply MOP and compost regularly, split nitrogen throughout growth.",
    "mango": "Apply NPK (100:100:100 g) per year of tree age; use farmyard manure annually.",
    "grapes": "High potassium and phosphorus; apply MOP and SSP; avoid excess nitrogen.",
    "watermelon": "Apply NPK in split doses; high potassium needed for fruit sweetness.",
    "muskmelon": "Similar to watermelon: split NPK doses; ensure potassium for sweetness.",
    "apple": "Apply NPK (urea, DAP, MOP) in early spring; add compost in autumn.",
    "orange": "Balanced NPK with extra potassium; apply micronutrients like zinc and iron.",
    "papaya": "High nitrogen during vegetative growth, more potassium during fruiting.",
    "coconut": "Apply NPK annually with magnesium; add compost or green manure.",
    "cotton": "Balanced NPK with extra potassium; organic manure improves soil texture.",
    "jute": "Apply nitrogen and phosphorus; use potash for better fibre quality.",
    "coffee": "Use organic compost and potassium sulfate; apply nitrogen after pruning."
}

# Crop Info
crop_info = {
    "rice": {
        "description": "Rice is a staple food crop grown in warm, humid climates, often in flooded fields.",
        "conditions": "Grows best in clayey loam soil, temperatures between 20–35°C, and high humidity.",
        "tips": "Maintain standing water in fields; use pest-resistant high-yield varieties."
    },
    "wheat": {
        "description": "Wheat is a major cereal crop grown in temperate regions worldwide.",
        "conditions": "Requires cool weather during early growth and dry conditions during harvest.",
        "tips": "Use certified seeds; apply nitrogen fertilizer in split doses."
    },
    "maize": {
        "description": "Maize (corn) is used as food, fodder, and industrial raw material.",
        "conditions": "Grows well in temperatures between 21–27°C with moderate rainfall.",
        "tips": "Ensure proper spacing for sunlight; control weeds early."
    },
    "chickpea": {
        "description": "Chickpea is a protein-rich legume used in many dishes.",
        "conditions": "Thrives in well-drained sandy loam soil in semi-arid climates.",
        "tips": "Rotate with cereals to improve soil fertility."
    },
    "kidneybeans": {
        "description": "Kidney beans are rich in protein and fiber, grown mainly in tropical areas.",
        "conditions": "Grows best in warm climates with well-drained soil.",
        "tips": "Avoid waterlogging; provide trellis support if needed."
    },
    "pigeonpeas": {
        "description": "Pigeon peas are a drought-resistant pulse crop.",
        "conditions": "Can grow in poor soils; prefers warm climates with moderate rainfall.",
        "tips": "Plant at the onset of rains; inter-crop with cereals."
    },
    "mothbeans": {
        "description": "Moth beans are drought-tolerant legumes grown in arid areas.",
        "conditions": "Survives in sandy soils with minimal water.",
        "tips": "Use early-maturing varieties in low rainfall areas."
    },
    "mungbean": {
        "description": "Mung beans are high in protein and used in soups, sprouts, and curries.",
        "conditions": "Grows well in warm climates with moderate rainfall.",
        "tips": "Avoid waterlogging; rotate with cereals to improve soil health."
    },
    "blackgram": {
        "description": "Black gram is a pulse crop used in various traditional dishes.",
        "conditions": "Thrives in warm, humid climates and loamy soils.",
        "tips": "Plant in well-drained soils; avoid excess irrigation."
    },
    "lentil": {
        "description": "Lentils are a major source of protein, especially in vegetarian diets.",
        "conditions": "Best grown in temperate climates with cool weather.",
        "tips": "Use disease-resistant varieties to reduce crop loss."
    },
    "pomegranate": {
        "description": "Pomegranate is a fruit crop valued for its sweet, tangy seeds.",
        "conditions": "Prefers dry climates with sandy loam soils.",
        "tips": "Prune regularly to maintain fruit quality."
    },
    "banana": {
        "description": "Bananas are tropical fruits grown for fresh consumption and export.",
        "conditions": "Requires warm, humid climates with rich loamy soil.",
        "tips": "Irrigate regularly; apply potassium-rich fertilizer."
    },
    "mango": {
        "description": "Mango is known as the king of fruits, popular for its sweetness.",
        "conditions": "Grows best in tropical/subtropical climates with well-drained soils.",
        "tips": "Prune after harvest; protect from frost."
    },
    "grapes": {
        "description": "Grapes are grown for fresh fruit, juice, and wine.",
        "conditions": "Requires warm, dry summers and well-drained soils.",
        "tips": "Train vines on trellises; manage pests like mealybugs."
    },
    "watermelon": {
        "description": "Watermelon is a refreshing summer fruit rich in water and vitamins.",
        "conditions": "Grows best in sandy loam soils with warm temperatures.",
        "tips": "Avoid over-irrigation; harvest when tendril near fruit turns brown."
    },
    "muskmelon": {
        "description": "Muskmelon is a sweet, fragrant fruit.",
        "conditions": "Prefers sandy loam soil and warm, dry climate.",
        "tips": "Ensure proper spacing; avoid waterlogging."
    },
    "apple": {
        "description": "Apple is a temperate fruit rich in vitamins and minerals.",
        "conditions": "Requires cold winters and moderate summers.",
        "tips": "Protect from frost during flowering; prune annually."
    },
    "orange": {
        "description": "Oranges are citrus fruits rich in vitamin C.",
        "conditions": "Thrives in subtropical climates with well-drained soils.",
        "tips": "Irrigate during dry spells; protect from citrus greening disease."
    },
    "papaya": {
        "description": "Papaya is a tropical fruit crop valued for its nutrition.",
        "conditions": "Prefers warm climates with sandy loam soils.",
        "tips": "Ensure male and female plants for fruit set; irrigate regularly."
    },
    "coconut": {
        "description": "Coconut palms are grown for food, oil, and fiber.",
        "conditions": "Requires sandy coastal soils and high humidity.",
        "tips": "Apply organic manure regularly; control rhinoceros beetle."
    },
    "cotton": {
        "description": "Cotton is a fiber crop used in textiles.",
        "conditions": "Needs long frost-free period and plenty of sunshine.",
        "tips": "Control bollworm pests; avoid waterlogging."
    },
    "jute": {
        "description": "Jute is a fiber crop used in making ropes and sacks.",
        "conditions": "Prefers warm, humid climates with alluvial soils.",
        "tips": "Harvest at flowering stage for best fiber quality."
    },
    "coffee": {
        "description": "Coffee is a beverage crop grown in tropical highlands.",
        "conditions": "Requires shade, cool temperatures, and well-drained soil.",
        "tips": "Control coffee berry borer; maintain shade trees."
    }
}

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
    
    # Initialize session state for weather so values persist
    if "temp" not in st.session_state: st.session_state.temp = 25.0
    if "hum" not in st.session_state: st.session_state.hum = 70.0
    if "rain" not in st.session_state: st.session_state.rain = 100.0

    # Weather & Soil Input Section
    col_soil, col_weather = st.columns([2, 1])
    
    with col_weather:
        st.write("### 🌦️ Local Weather")
        city = st.text_input("Enter City", "Hyderabad")
        # Buttons in 2026 are responsive by default
        if st.button("Fetch Live Weather"):
            t, h, r, err = get_weather(city)
            if not err:
                st.session_state.temp, st.session_state.hum, st.session_state.rain = t, h, r
                st.rerun() # Refresh to update sliders with new weather data
            else: 
                st.error(err)
        
        st.metric("Temperature", f"{st.session_state.temp}°C")
        st.metric("Humidity", f"{st.session_state.hum}%")

    with col_soil:
        st.write("### 🧪 Soil & Environmental Conditions")
        n_col, p_col, k_col = st.columns(3)
        N = n_col.number_input("Nitrogen (N)", 0, 150, 50)
        P = p_col.number_input("Phosphorus (P)", 0, 150, 50)
        K = k_col.number_input("Potassium (K)", 0, 150, 50)
        
        ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5)
        
        # Environmental inputs linked to session state
        temp = st.number_input("Temperature (°C)", -10.0, 50.0, float(st.session_state.temp))
        hum = st.number_input("Humidity (%)", 0.0, 100.0, float(st.session_state.hum))
        rain = st.number_input("Rainfall (mm)", 0.0, 500.0, float(st.session_state.rain))

    # --- CENTERED RECOMMENDATION BUTTON ---
    st.write("---")
    rb1, rb2, rb3 = st.columns([1, 1, 1])
    with rb2:
        if st.button("Identify Ideal Crop"):
            if crop_model:
                features = np.array([[N, P, K, temp, hum, ph, rain]])
                prediction = crop_model.predict(features)
                crop = label_mapping[int(prediction[0])]
                
                st.balloons()
                st.markdown(f"""
                    <div class='prediction-card'>
                        <h2>Best Crop: {crop.capitalize()}</h2>
                        <p>Based on current soil and weather data</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display Crop-Specific Advice
                if crop in crop_info:
                    with st.expander("📖 Crop Details & Growing Conditions"):
                        st.write(crop_info[crop]['description'])
                        st.info(crop_info[crop]['conditions'])
                
                if crop in fertilizer_advice:
                    st.success(f"**Fertilizer Advice:** {fertilizer_advice[crop]}")
            else:
                st.error("Crop Recommendation Model not loaded. Check your joblib file.")

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
