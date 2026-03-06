
import streamlit as st
import joblib
import numpy as np
import requests
import os

# --- 1. BOILERPLATE & INITIALIZATION ---
st.set_page_config(page_title="Crop Recommendation", layout="wide")

# Initialize Session State early to prevent widget rendering errors
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
