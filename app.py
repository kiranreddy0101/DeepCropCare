import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
from io import BytesIO
import cv2
import joblib
import requests
import os

# ---------------------- GRAD-CAM FUNCTIONS ---------------------- #
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
    return overlay_img

# ---------------------- STREAMLIT PAGE CONFIG ---------------------- #
st.set_page_config(page_title="DeepCropCare - Disease Detection", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.prediction-card {
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
    font-size: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    background-color: #1e1e1e;
    color: #ffffff;
}
.fertilizer-card {
    padding: 15px;
    border-radius: 12px;
    margin-top: 15px;
    text-align: center;
    background: rgba(0, 255, 140, 0.1);
    border: 2px solid rgba(0, 255, 140, 0.4);
    color: #ffffff;
}
h1, h2, h3, p { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------------- DATA & MODELS ---------------------- #
@st.cache_resource
def load_trained_model():
    return load_model("plant_disease_model_final4.h5")

model = load_trained_model()

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

# Fertilizer Mapping
fertilizer_map = {
    'Apple___Apple_scab': 'Use copper-based fungicides',
    'Apple___Black_rot': 'Apply sulfur sprays or captan',
    'Apple___Cedar_apple_rust': 'Use myclobutanil or mancozeb',
    'Bitter Gourd__Downy_mildew': 'Use mancozeb or copper oxychloride spray',
    'Bitter Gourd__Fusarium_wilt': 'Apply Trichoderma and maintain soil pH; avoid overwatering',
    'Bitter Gourd__Mosaic_virus': 'Remove infected plants; control whiteflies with neem oil',
    'Bottle gourd__Anthracnose': 'Apply chlorothalonil or mancozeb',
    'Bottle gourd__Downey_mildew': 'Use metalaxyl or copper-based fungicides',
    'Cauliflower__Black_Rot': 'Apply copper-based bactericide; ensure crop rotation',
    'Cauliflower__Downy_mildew': 'Use foliar sprays with mancozeb or metalaxyl',
    'Cherry___Powdery_mildew': 'Spray with sulfur or neem oil',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Use nitrogen-balanced fertilizers',
    'Corn___Common_rust': 'Apply fungicides like propiconazole',
    'Corn___Northern_Leaf_Blight': 'Use mancozeb or chlorothalonil',
    'Cucumber__Anthracnose_lesions': 'Use fungicides like chlorothalonil or copper hydroxide',
    'Cucumber__Downy_mildew': 'Apply mancozeb or fosetyl-aluminum',
    'Eggplant_Cercopora_leaf_spot': 'Use mancozeb or zineb sprays',
    'Eggplant_begomovirus': 'Control whiteflies; use resistant varieties',
    'Eggplant_verticillium_wilt': 'Use solarized soil; apply bio-fungicide like Trichoderma',
    'Grape___Black_rot': 'Spray with captan or mancozeb',
    'Grape___Esca_(Black_Measles)': 'Avoid excess nitrogen; apply phosphorus',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Use Bordeaux mixture',
    'Guava_Phytopthora': 'Apply metalaxyl; improve drainage',
    'Guava_Red_rust': 'Spray with copper oxychloride and urea solution',
    'Guava_Scab': 'Use copper-based fungicide; prune infected twigs',
    'Guava_Styler_and_Root': 'Apply zinc and boron; use Trichoderma at root zone',
    'Orange___Haunglongbing_(Citrus_greening)': 'Apply zinc & manganese-rich foliar sprays',
    'Paddy_Bacterial_leaf_blight': 'Apply bleaching powder; spray streptocycline',
    'Paddy_Brown_spot': 'Use potassium-rich fertilizer; spray with mancozeb',
    'Paddy_Leaf_smut': 'Apply fungicides like propiconazole',
    'Peach___Bacterial_spot': 'Use oxytetracycline sprays',
    'Pepper_bell___Bacterial_spot': 'Spray copper-based bactericides',
    'Potato___Early_blight': 'Use azoxystrobin and increase potassium',
    'Potato___Late_blight': 'Apply metalaxyl-M fungicide',
    'Squash___Powdery_mildew': 'Use neem oil or sulfur-based spray',
    'Strawberry___Leaf_scorch': 'Apply copper-based fungicide',
    'Sugarcane_Mosaic': 'Remove infected plants; control aphids with imidacloprid',
    'Sugarcane_RedRot': 'Use resistant varieties and proper field drainage',
    'Sugarcane_Rust': 'Spray fungicides like mancozeb or propiconazole',
    'Sugarcane_Yellow': 'Apply micronutrients (zinc, iron); foliar spray of ferrous sulfate',
    'Tomato___Bacterial_spot': 'Use copper sprays, avoid overhead watering',
    'Tomato___Early_blight': 'Apply chlorothalonil',
    'Tomato___Late_blight': 'Spray with mancozeb or copper-based fungicide',
    'Tomato___Leaf_Mold': 'Increase airflow and use fungicides',
    'Tomato___Septoria_leaf_spot': 'Apply fungicide with chlorothalonil',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use insecticidal soap or neem oil',
    'Tomato___Target_Spot': 'Apply fungicides like pyraclostrobin',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Use resistant varieties; spray imidacloprid',
    'Tomato___Tomato_mosaic_virus': 'Use resistant cultivars and disinfect tools',
    'Wheat_leaf_leaf_stripe_rust': 'Apply propiconazole; avoid excessive nitrogen',
    'Wheatleaf_septoria': 'Spray fungicide with chlorothalonil or tebuconazole'
}

# ---------------------- MAIN APP ---------------------- #
tab1, tab2, tab3= st.tabs(["🌱 Disease Detection", "📘 Info","🌱 CropRecommendation"])

with tab1:
    st.markdown("## 🌿 Plant Disease Analysis")
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display Image
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_data = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{img_data}" width="350" style="border-radius: 15px;"/></div>', unsafe_allow_html=True)

        # Inference
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner('Analyzing...'):
            prediction = model.predict(img_array)
            idx = np.argmax(prediction)
            predicted_class = class_names[idx]
            confidence = np.max(prediction) * 100

        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='prediction-card'>🔎 <b>Result:</b><br>{predicted_class.replace('___', ' - ')}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='prediction-card'>🎯 <b>Confidence:</b><br>{confidence:.2f}%</div>", unsafe_allow_html=True)

        # Fertilizer Advisory Logic
        if "healthy" in predicted_class.lower():
            st.success("✅ The leaf is healthy. Continue standard care and maintenance!")
        elif predicted_class in fertilizer_map:
            st.markdown(f"<div class='fertilizer-card'>🧪 <b>Fertilizer Advisory:</b><br>{fertilizer_map[predicted_class]}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No specific fertilizer recommendation found for this condition.")

        # Grad-CAM
        st.markdown("### 📊 Grad-CAM Visualization")
        heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name="Conv_1")
        overlay_img = overlay_gradcam(img_resized, heatmap)
        st.image(overlay_img, caption="AI focus area for diagnosis", use_container_width=True)

with tab2:
    st.markdown("## 📘 System Information")
    st.write("DeepCropCare bridges the gap between diagnosis and prevention through high-accuracy AI models and real-time advisory logic.")


with tab3:
      if "weather_temp" not in st.session_state: 
    st.session_state.weather_temp = 25.0
if "weather_hum" not in st.session_state: 
    st.session_state.weather_hum = 70.0
if "weather_rain" not in st.session_state: 
    st.session_state.weather_rain = 100.0

# --- 2. MODEL LOADING (WITH FALLBACK) ---
@st.cache_resource
def load_model(file_name):
    """Safely loads the model from current or Colab directory."""
    # Check current directory and /content/ (standard for Colab)
    possible_paths = [file_name, os.path.join("/content", file_name)]
    
    for p in possible_paths:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception as e:
                st.error(f"❌ Error loading model at {p}: {e}")
    return None

model = load_model("rf_crop_recommendation.joblib")

# --- 3. DATA MAPPINGS ---
label_mapping = {
    0: "rice", 1: "wheat", 2: "maize", 3: "chickpea", 4: "kidneybeans",
    5: "pigeonpeas", 6: "mothbeans", 7: "mungbean", 8: "blackgram", 9: "lentil",
    10: "pomegranate", 11: "banana", 12: "mango", 13: "grapes", 14: "watermelon",
    15: "muskmelon", 16: "apple", 17: "orange", 18: "papaya", 19: "coconut",
    20: "cotton", 21: "jute", 22: "coffee"
}

fertilizer_advice = {
    "rice": "Apply Urea, DAP, and MOP in split doses; ensure timely irrigation.",
    "wheat": "Use nitrogen-rich urea and phosphorus (DAP) before sowing; top-dress at tillering stage.",
    "maize": "Balanced NPK application with nitrogen split at 30 & 45 days; use zinc sulphate if deficient.",
    "chickpea": "Use Rhizobium inoculant and phosphorus-rich fertilizers; avoid excess nitrogen.",
    "kidneybeans": "Use balanced NPK with high phosphorus; apply well-rotted manure.",
    "pigeonpeas": "Apply DAP or SSP at sowing; use organic compost.",
    "mothbeans": "Use moderate NPK with emphasis on phosphorus; apply farmyard manure.",
    "mungbean": "Apply Rhizobium culture and balanced NPK.",
    "blackgram": "Use Rhizobium and phosphate solubilizing bacteria; apply SSP at sowing.",
    "lentil": "Apply phosphorus and potassium; treat seeds with Rhizobium.",
    "pomegranate": "Apply high potassium (MOP) during fruit development.",
    "banana": "High potassium requirement; apply MOP and compost regularly.",
    "mango": "Apply NPK (100:100:100 g) per year of age; use farmyard manure.",
    "grapes": "High potassium and phosphorus; train vines on trellises.",
    "watermelon": "Apply NPK in split doses; high potassium for sweetness.",
    "muskmelon": "Split NPK doses; ensure potassium for sweetness.",
    "apple": "Apply urea, DAP, MOP in early spring; add compost in autumn.",
    "orange": "Balanced NPK with extra potassium; apply micronutrients like zinc.",
    "papaya": "High nitrogen during growth, more potassium during fruiting.",
    "coconut": "Apply NPK annually with magnesium; add green manure.",
    "cotton": "Balanced NPK with extra potassium; organic manure improves texture.",
    "jute": "Apply nitrogen and phosphorus; use potash for quality fibre.",
    "coffee": "Requires high organic matter; balanced NPK with emphasis on Nitrogen."
}

# Crop Info (truncated for brevity)
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

# --- 4. WEATHER UTILITY ---
@st.cache_data(ttl=600)
def get_weather(city_name):
    """Fetches real-time weather data and extracts values safely."""
    API_KEY = "8c3a497f31607fe66be1f23c65538904"
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    try:
        q = city_name if "," in city_name else f"{city_name},IN"
        params = {"q": q, "appid": API_KEY, "units": "metric"}
        response = requests.get(BASE_URL, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            rainfall = data.get("rain", {}).get("1h", 0) # Handle missing rain key
            return temp, humidity, rainfall, None
    except Exception as e:
        return None, None, None, str(e)
    return None, None, None, "City not found"

# --- 5. UI LAYOUT ---
st.title("🌾 Smart Crop Recommendation System")

with st.sidebar:
    st.header("🌍 Weather Integration")
    city = st.text_input("Enter City", value="Hyderabad")
    if st.button("Fetch Live Weather"):
        t, h, r, err = get_weather(city)
        if not err:
            st.session_state.weather_temp = float(t)
            st.session_state.weather_hum = float(h)
            st.session_state.weather_rain = float(r)
            st.success(f"Updated for {city}")
        else:
            st.error(f"Weather error: {err}")

# --- 6. INPUT FIELDS ---
st.subheader("📊 Input Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, key="weather_temp")
with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, key="weather_hum")
with col3:
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, key="weather_rain")

st.subheader("🧪 Soil Nutrients & pH")
col4, col5, col6, col7 = st.columns(4)

with col4: N = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
with col5: P = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0)
with col6: K = st.number_input("Potassium (K)", 0.0, 200.0, 50.0)
with col7: ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

# --- 7. PREDICTION LOGIC ---
st.divider()
if st.button("🌱 Recommend Optimal Crop", use_container_width=True):
    if model is not None:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        try:
            prediction = model.predict(features)
            crop_name = label_mapping[int(prediction[0])]
            
            st.success(f"### 🎉 Recommended Crop: **{crop_name.upper()}**")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if crop_name in crop_info:
                    st.info("📜 **Crop Details**")
                    st.write(crop_info[crop_name]["description"])
                    st.write(f"**Conditions:** {crop_info[crop_name]['conditions']}")
            with res_col2:
                if crop_name in fertilizer_advice:
                    st.warning("💡 **Fertilizer Advisory**")
                    st.write(fertilizer_advice[crop_name])
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model not loaded. Ensure 'rf_crop_recommendation.joblib' is in the app directory.")

    
