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

# --- 1. MULTI-LANGUAGE DICTIONARY ---
lang_dict = {
    "English": {
        "title": "DeepCropCare",
        "disease_tab": "🔍 Disease Detection",
        "crop_tab": "🌾 Crop Recommendation",
        "info_tab": "📘 Project Info",
        "upload_label": "Upload leaf image",
        "btn_analyze": "Run Diagnostic Analysis",
        "soil_header": "🧪 Soil Parameters",
        "weather_header": "🌦️ Local Weather",
        "village_input": "Enter Village & District",
        "btn_fetch": "Fetch Live Weather",
        "btn_recommend": "Recommend Best Crop",
        "analysis_done": "Analysis Complete!",
        "optimal_cond": "🔍 Optimal Conditions:",
        "fert_advice": "🧪 Fertilizer & Care Advice"
    },
    "Telugu": {
        "title": "డీప్ క్రాప్ కేర్",
        "disease_tab": "🔍 వ్యాధి నిర్ధారణ",
        "crop_tab": "🌾 పంట సిఫార్సు",
        "info_tab": "📘 ప్రాజెక్ట్ సమాచారం",
        "upload_label": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "btn_analyze": "రోగ నిర్ధారణ విశ్లేషణను ప్రారంభించండి",
        "soil_header": "🧪 నేల పారామితులు",
        "weather_header": "🌦️ స్థానిక వాతావరణం",
        "village_input": "గ్రామం మరియు జిల్లాను నమోదు చేయండి",
        "btn_fetch": "ప్రత్యక్ష వాతావరణాన్ని పొందండి",
        "btn_recommend": "ఉత్తమ పంటను సిఫార్సు చేయండి",
        "analysis_done": "విశ్లేషణ పూర్తయింది!",
        "optimal_cond": "🔍 సరైన పరిస్థితులు:",
        "fert_advice": "🧪 ఎరువులు & సంరక్షణ సలహా"
    },
    "Hindi": {
        "title": "डीप क्रॉप केयर",
        "disease_tab": "🔍 रोग पहचान",
        "crop_tab": "🌾 फसल अनुशंसा",
        "info_tab": "📘 प्रोजेक्ट जानकारी",
        "upload_label": "पत्ती की छवि अपलोड करें",
        "btn_analyze": "नैदानिक विश्लेषण चलाएं",
        "soil_header": "🧪 मिट्टी के मापदंड",
        "weather_header": "🌦️ स्थानीय मौसम",
        "village_input": "गांव और जिला दर्ज करें",
        "btn_fetch": "लाइव मौसम प्राप्त करें",
        "btn_recommend": "सर्वोत्तम फसल की सिफारिश करें",
        "analysis_done": "विश्लेषण पूरा हुआ!",
        "optimal_cond": "🔍 इष्टतम स्थितियां:",
        "fert_advice": "🧪 उर्वरक और देखभाल सलाह"
    }
}

# --- 2. CONFIG & STYLING ---
st.set_page_config(page_title="DeepCropCare", layout="wide",initial_sidebar_state="expanded")

# Sidebar Language Selection
with st.sidebar:
    st.markdown("### 🌐 Language Settings")
    selected_lang = st.selectbox("Choose Language / భాషను ఎంచుకోండి / भाषा चुनें", ["English", "Telugu", "Hindi"])
    L = lang_dict[selected_lang] # Shortcut variable

st.markdown(f"""
    <style>
    /* 1. CLEAN TOP SPACING - UPDATED */
    header {
        visibility: visible !important; /* This brings back the top-right menu */
        background: rgba(0,0,0,0) !important; /* Makes the header background transparent */
    }
    
    /* This removes only the "Deploy" button but keeps the menu */
    .stAppDeployButton {display:none;} 
    
    .block-container {
        padding-top: 2rem !important; /* Give a little space for the menu */
        margin-top: -2rem !important; 
        max-width: 95%; 
    }
    header {{visibility: hidden;}}
    .stAppDeployButton {{display:none;}}
    .block-container {{ padding-top: 1.5rem !important; margin-top: -4rem !important; max-width: 95%; }}
    .top-header {{ text-align: center; padding-bottom: 1rem; }}
    div.stButton > button {{
        width: 100% !important; white-space: nowrap !important; font-weight: bold !important;
        background-color: #28a745 !important; color: white !important; border-radius: 10px !important;
    }}
    .stApp {{ background: radial-gradient(circle at top right, #1a2e1a, #0e1117); background-attachment: fixed; }}
    .prediction-card {{ 
        background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(12px); border-radius: 20px; 
        padding: 30px; text-align: center; margin: 20px 0px; border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8); border-bottom: 4px solid #28a745;
    }}
    .prediction-card h2 {{ color: #28a745 !important; font-weight: 700 !important; }}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown(f"""
    <div class="top-header">
        <h1 style="font-size: 3.5rem; color: #28a745; margin-bottom: 0;">🌱 {L['title']}</h1>
        <p style="font-size: 1.1rem; color: #a3a3a3; margin-top: -5px;">Precision AI for Plant Health & Smarter Yields</p>
    </div>
""", unsafe_allow_html=True)

# --- FUNCTIONS (Grad-CAM, Weather, Resources) ---
# [Keep your get_gradcam_heatmap, overlay_gradcam, get_weather, and load_resources here as they were]

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predictions = tf.squeeze(predictions)
        if pred_index is None: pred_index = tf.argmax(predictions)
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

@st.cache_resource
def load_resources():
    try: d_model = load_model("plant_disease_model_final4.h5", compile=False)
    except: d_model = None
    detected_name = None
    if d_model:
        for layer in reversed(d_model.layers):
            if len(layer.output.shape) == 4 and not any(x in layer.name.lower() for x in ['flatten', 'gap', 'pool']):
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

# [Keep your class_names, fertilizer_map, label_mapping, fertilizer_advice, crop_info here]
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

# --- TABS ---
tab1, tab2, tab3 = st.tabs([L["disease_tab"], L["crop_tab"], L["info_tab"]])

with tab1:
    st.markdown(f"## {L['disease_tab']}")
    uploaded_file = st.file_uploader(L["upload_label"], type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col_s1, col_img, col_s2 = st.columns([1, 0.8, 1])
        with col_img:
            st.image(image, caption="Specimen", use_container_width=True)
        
        _, center_col, _ = st.columns([1, 1, 1])
        with center_col:
            run_btn = st.button(L["btn_analyze"], use_container_width=True)
        
        if run_btn: 
            progress_bar = st.progress(0)
            for pc in range(100):
                time.sleep(0.001)
                progress_bar.progress(pc + 1)
    
            with st.spinner("🧠 Scanning..."):
                if disease_model:
                    img_resized = image.resize((224, 224))
                    img_arr = img_to_array(img_resized) / 255.0
                    img_arr = np.expand_dims(img_arr, axis=0)
                    prediction = disease_model.predict(img_arr)
                    idx = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                    full_class_name = class_names[idx]
                    p_class_display = full_class_name.replace('___', ' ').replace('_', ' ')
                    
                    progress_bar.empty()
                    st.markdown(f"<h3 style='text-align: center;'>{L['analysis_done']}</h3>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class='prediction-card'>
                            <h2 style='margin:0;'>{p_class_display}</h2>
                            <h3 style='color: #28a745; margin:0;'>Confidence: {confidence:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if full_class_name in fertilizer_map:
                        st.info(f"**💡 Recommended Action:** {fertilizer_map[full_class_name]}")
                    
                    if "healthy" not in full_class_name.lower() and detected_conv_name:
                        heatmap = get_gradcam_heatmap(disease_model, img_arr, detected_conv_name)
                        overlay = overlay_gradcam(img_resized, heatmap)
                        col_a, col_b = st.columns(2)
                        with col_a: st.image(img_resized, caption="Original", use_container_width=True)
                        with col_b: st.image(overlay, caption="Heatmap", use_container_width=True)
                else:
                    st.error("Model Error")

with tab2:
    st.markdown(f"## {L['crop_tab']}")
    if "weather_temp" not in st.session_state: st.session_state.weather_temp = 25.0
    if "weather_hum" not in st.session_state: st.session_state.weather_hum = 70.0

    col_soil, col_weather = st.columns([1.5, 1])
    with col_soil:
        st.write(f"### {L['soil_header']}")
        n1, p1, k1 = st.columns(3)
        N = n1.number_input("N", 0, 200, 50)
        P = p1.number_input("P", 0, 200, 50)
        K = k1.number_input("K", 0, 200, 50)
        ph = st.slider("pH", 0.0, 14.0, 6.5)
        rain = st.number_input("Rainfall (mm)", 0.0, 1000.0, 100.0)

    with col_weather:
        st.write(f"### {L['weather_header']}")
        city = st.text_input(L["village_input"], "Hyderabad")
        if st.button(L["btn_fetch"], use_container_width=True):
            t, h, err = get_weather(city)
            if not err:
                st.session_state.weather_temp, st.session_state.weather_hum = float(t), float(h)
                st.success(f"📍 {city}")
        
        st.session_state.weather_temp = st.number_input("Temp", value=float(st.session_state.weather_temp))
        st.session_state.weather_hum = st.number_input("Humidity", value=float(st.session_state.weather_hum))

    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        predict_btn = st.button(L["btn_recommend"], use_container_width=True)

    if predict_btn:
        if crop_model:
            features = np.array([[N, P, K, st.session_state.weather_temp, st.session_state.weather_hum, ph, rain]])
            crop = label_mapping[crop_model.predict(features)[0]]
            
            st.markdown(f"<div class='prediction-card'><h2>🌱 Recommended: {crop.upper()}</h2></div>", unsafe_allow_html=True)
            inf1, inf2 = st.columns(2)
            with inf1:
                st.markdown(f"### 📖 Description")
                st.markdown(f"<div style='background: #1a1c23; padding: 15px; border-radius: 10px;'>{crop_info[crop]['description']}</div>", unsafe_allow_html=True)
                st.markdown(f"<br><b>{L['optimal_cond']}</b><br>{crop_info[crop]['conditions']}", unsafe_allow_html=True)
            with inf2:
                st.markdown(f"### {L['fert_advice']}")
                st.warning(fertilizer_advice[crop])
                st.success(f"Pro-Tip: {crop_info[crop]['tips']}")

with tab3:
    st.markdown("## 📘 About DeepCropCare")
    
    st.markdown("""
    ### 🚀 The Mission
    **DeepCropCare** is a cutting-edge agricultural platform designed to empower farmers with data-driven insights. 
    By merging **Deep Learning** (for leaf diagnostics) and **Machine Learning** (for crop suitability), 
    we provide a 360-degree view of your farm's potential and health.
    """)

    st.divider()

    col_cv, col_ml = st.columns(2)
    with col_cv:
        st.markdown("#### 🧠 Computer Vision")
        st.write("""
        Using **Convolutional Neural Networks (CNN)**, the system analyzes leaf texture and 
        patterns to detect 38 different plant-disease states.
        """)
        
    with col_ml:
        st.markdown("#### 📈 Predictive Analytics")
        st.write("""
        Our **Random Forest** engine processes soil NPK levels and weather data to 
        recommend the ideal crop for your land.
        """)

    # Soil Chemistry Guide
    with st.expander("🧪 Understanding Soil Parameters (N-P-K)"):
        st.write("""
        - **Nitrogen (N):** Essential for leaf growth and green color.
        - **Phosphorus (P):** Critical for root development and flower/seed production.
        - **Potassium (K):** Helps with overall plant health and disease resistance.
        """)
        

    st.divider()

    # Grad-CAM explanation
    st.markdown("### 🎯 Interpretability: How the AI 'Sees'")
    
    
    st.info(f"**Target Diagnostic Layer:** `{detected_conv_name}`. The heatmap highlights areas the AI identified as diseased.")

    st.caption("DeepCropCare v1.0 | 2026 Agricultural Innovation")
