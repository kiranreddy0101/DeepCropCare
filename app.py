import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
from io import BytesIO
import cv2

# ---------------------- INITIAL SETUP ---------------------- #
st.set_page_config(page_title="DeepCropCare AI", layout="centered", initial_sidebar_state="collapsed")

# ---------------------- MODERN UI CUSTOMIZATION ---------------------- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Dark Theme Background */
    .stApp {
        background-color: #0f1116;
        color: #e6edf3;
    }

    /* Modern Action Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 16px;
        height: 80px;
        background-color: #161b22;
        color: #ffffff;
        border: 1px solid #30363d;
        font-size: 18px;
        font-weight: 600;
        transition: 0.3s all ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stButton>button:hover {
        border-color: #00ff8c;
        background-color: #00ff8c10;
        transform: translateY(-2px);
    }

    /* Results Dashboard */
    .dashboard-card {
        background: #161b22;
        padding: 24px;
        border-radius: 24px;
        border: 1px solid #30363d;
        margin: 20px 0;
        text-align: center;
    }
    
    .status-healthy { color: #00ff8c; font-size: 28px; font-weight: 800; margin-bottom: 10px; }
    .status-disease { color: #ff4b4b; font-size: 28px; font-weight: 800; margin-bottom: 10px; }
    
    .advisory-card {
        background: rgba(0, 255, 140, 0.08);
        border-left: 6px solid #00ff8c;
        padding: 16px;
        border-radius: 12px;
        text-align: left;
        margin-top: 15px;
    }

    /* Hide standard Streamlit elements for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------- CORE ML FUNCTIONS ---------------------- #
@st.cache_resource
def load_trained_model():
    return load_model("plant_disease_model_final4.h5")

def get_gradcam_heatmap(model, img_array, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, tf.argmax(predictions[0])]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(original_img), 0.6, heatmap, 0.4, 0)
    return superimposed_img

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

# ---------------------- MAIN APP UI ---------------------- #
model = load_trained_model()

st.markdown("<h1 style='text-align: center;'>🌿 DeepCropCare AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.7;'>Instant Plant Disease Diagnosis & Fertilizer Advisory</p>", unsafe_allow_html=True)

# Interaction Hub
if 'active_mode' not in st.session_state:
    st.session_state.active_mode = None

col1, col2 = st.columns(2)
with col1:
    if st.button("📁 Upload Leaf"):
        st.session_state.active_mode = 'upload'
with col2:
    if st.button("📷 Scan with Camera"):
        st.session_state.active_mode = 'camera'

# Image Input Logic
file_source = None
if st.session_state.active_mode == 'upload':
    file_source = st.file_uploader("Select image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
elif st.session_state.active_mode == 'camera':
    file_source = st.camera_input("Capture leaf photo", label_visibility="collapsed")

# ---------------------- ANALYSIS ENGINE ---------------------- #
if file_source:
    raw_img = Image.open(file_source).convert("RGB")
    
    # Pre-process
    target_size = (224, 224)
    img_resized = raw_img.resize(target_size)
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🧠 AI is analyzing the specimen..."):
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds)
        result = class_names[pred_idx]
        score = np.max(preds) * 100

    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    
    if "healthy" in result.lower():
        st.markdown(f"<div class='status-healthy'>✅ Healthy Specimen</div>", unsafe_allow_html=True)
        st.write(f"The model is {score:.1f}% confident that this {result.split('___')[0]} leaf is healthy.")
        st.balloons()
    else:
        st.markdown(f"<div class='status-disease'>🚨 {result.replace('___', ' - ')} Detected</div>", unsafe_allow_html=True)
        
        # Advisory Logic
        if result in fertilizer_map:
            st.markdown(f"""
            <div class='advisory-card'>
                <span style='color: #00ff8c; font-weight: 800;'>🧪 FERTILIZER ADVISORY</span><br>
                <p style='margin-top: 8px;'>{fertilizer_map[result]}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Treatment data for this specific class is currently being updated.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Visualization Section
    st.divider()
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.image(raw_img, caption="Source Image", use_container_width=True)
    with v_col2:
        heatmap = get_gradcam_heatmap(model, img_array)
        grad_cam_img = overlay_heatmap(img_resized, heatmap)
        st.image(grad_cam_img, caption="AI Symptom Localization (Grad-CAM)", use_container_width=True)

st.markdown("<br><p style='text-align: center; opacity: 0.4; font-size: 12px;'>© 2026 DeepCropCare Systems</p>", unsafe_allow_html=True)
