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


def prettify_label(label):
    return label.replace("___", " - ").replace("__", " - ").replace("_", " ")


# ---------------------- STREAMLIT PAGE CONFIG ---------------------- #
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Light Mode */
    @media (prefers-color-scheme: light) {
        body, html, [class*="css"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .prediction-card {
            background-color: #f0f0f0;
            color: #000000;
        }
        .sidebar-text {
            color: #000000 !important;
        }
    }

    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
        body, html, [class*="css"] {
            background-color: #121212 !important;
            color: #ffffff !important;
        }
        .prediction-card {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .sidebar-text {
            color: #ffffff !important;
        }
    }

    .prediction-card {
        padding: 12px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 16px;
        text-align: center;
    }

    h1, h3, p {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------- LOAD MODEL ---------------------- #
@st.cache_resource
def load_trained_model():
    return load_model("plant_disease_model_final4.h5")


model = load_trained_model()


# ---------------------- CLASS LABELS ---------------------- #
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


# ---------------------- FERTILIZER MAP ---------------------- #
fertilizer_map = {
    'Apple___Apple_scab': 'Use copper-based fungicides',
    'Apple___Black_rot': 'Apply sulfur sprays or captan',
    'Apple___Cedar_apple_rust': 'Use myclobutanil or mancozeb',
    'Apple___healthy': 'No fertilizer needed',
    'Background_without_leaves': 'N/A',
    'Bitter Gourd__Downy_mildew': 'Use mancozeb or copper oxychloride spray',
    'Bitter Gourd__Fusarium_wilt': 'Apply Trichoderma and maintain soil pH; avoid overwatering',
    'Bitter Gourd__Fresh_leaf': 'Use balanced NPK (10-10-10)',
    'Bitter Gourd__Mosaic_virus': 'Remove infected plants; control whiteflies with neem oil',
    'Blueberry___healthy': 'Use ammonium sulfate',
    'Bottle gourd__Anthracnose': 'Apply chlorothalonil or mancozeb',
    'Bottle gourd__Downey_mildew': 'Use metalaxyl or copper-based fungicides',
    'Bottle gourd__Fresh_leaf': 'Apply compost or organic fertilizer with potassium',
    'Cauliflower__Black_Rot': 'Apply copper-based bactericide; ensure crop rotation',
    'Cauliflower__Downy_mildew': 'Use foliar sprays with mancozeb or metalaxyl',
    'Cauliflower__Fresh_leaf': 'Use nitrogen-rich fertilizer for leafy growth',
    'Cherry___Powdery_mildew': 'Spray with sulfur or neem oil',
    'Cherry___healthy': 'Fertilize with potassium-rich mix',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Use nitrogen-balanced fertilizers',
    'Corn___Common_rust': 'Apply fungicides like propiconazole',
    'Corn___Northern_Leaf_Blight': 'Use mancozeb or chlorothalonil',
    'Corn___healthy': 'Apply NPK (20-20-20) for growth',
    'Cucumber__Anthracnose_lesions': 'Use fungicides like chlorothalonil or copper hydroxide',
    'Cucumber__Downy_mildew': 'Apply mancozeb or fosetyl-aluminum',
    'Cucumber__Fresh_leaf': 'Apply organic compost or vermicompost',
    'Eggplant_Cercopora_leaf_spot': 'Use mancozeb or zineb sprays',
    'Eggplant_begomovirus': 'Control whiteflies; use resistant varieties',
    'Eggplant_fresh_leaf': 'Use compost and potassium-rich fertilizer',
    'Eggplant_verticillium_wilt': 'Use solarized soil; apply bio-fungicide like Trichoderma',
    'Grape___Black_rot': 'Spray with captan or mancozeb',
    'Grape___Esca_(Black_Measles)': 'Avoid excess nitrogen; apply phosphorus',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Use Bordeaux mixture',
    'Grape___healthy': 'Balanced NPK mix',
    'Guava_Healthy': 'Apply NPK 6:6:6 monthly during growing season',
    'Guava_Phytopthora': 'Apply metalaxyl; improve drainage',
    'Guava_Red_rust': 'Spray with copper oxychloride and urea solution',
    'Guava_Scab': 'Use copper-based fungicide; prune infected twigs',
    'Guava_Styler_and_Root': 'Apply zinc and boron; use Trichoderma at root zone',
    'Orange___Haunglongbing_(Citrus_greening)': 'Apply zinc & manganese-rich foliar sprays',
    'Paddy_Bacterial_leaf_blight': 'Apply bleaching powder; spray streptocycline',
    'Paddy_Brown_spot': 'Use potassium-rich fertilizer; spray with mancozeb',
    'Paddy_Leaf_smut': 'Apply fungicides like propiconazole',
    'Peach___Bacterial_spot': 'Use oxytetracycline sprays',
    'Peach___healthy': 'Use low-nitrogen fertilizer',
    'Pepper_bell___Bacterial_spot': 'Spray copper-based bactericides',
    'Pepper_bell___healthy': 'Use 5-10-10 fertilizer',
    'Potato___Early_blight': 'Use azoxystrobin and increase potassium',
    'Potato___Late_blight': 'Apply metalaxyl-M fungicide',
    'Potato___healthy': 'Use nitrogen-rich compost',
    'Raspberry___healthy': 'Use 10-10-10 fertilizer',
    'Soybean___healthy': 'Use rhizobium inoculants + phosphorus',
    'Squash___Powdery_mildew': 'Use neem oil or sulfur-based spray',
    'Strawberry___Leaf_scorch': 'Apply copper-based fungicide',
    'Strawberry___healthy': 'Use phosphorus-heavy fertilizer',
    'Sugarcane_Healthy': 'Apply NPK (25:50:75) during planting and top dressing',
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
    'Tomato___healthy': 'Use balanced NPK fertilizer (10-10-10)',
    'Wheat_Healthy': 'Use DAP or balanced NPK during sowing and tillering',
    'Wheat_leaf_leaf_stripe_rust': 'Apply propiconazole; avoid excessive nitrogen',
    'Wheatleaf_septoria': 'Spray fungicide with chlorothalonil or tebuconazole'
}


# ---------------------- DISEASE INFO (ALL CLASSES AUTO) ---------------------- #
# This will automatically provide info for every label.
# If a label is not mapped specifically, it gives generic symptoms/prevention.
disease_info = {}

for cls in class_names:
    if "healthy" in cls.lower():
        disease_info[cls] = {
            "symptoms": "No symptoms detected. Leaf appears healthy.",
            "prevention": "Maintain balanced irrigation, fertilization, and pest monitoring."
        }
    elif "background" in cls.lower():
        disease_info[cls] = {
            "symptoms": "No plant leaf detected clearly.",
            "prevention": "Upload a clear close-up leaf image with good lighting."
        }
    else:
        disease_info[cls] = {
            "symptoms": "Leaf may show spots, discoloration, wilting, curling or fungal growth depending on disease type.",
            "prevention": "Avoid overhead watering, ensure airflow, remove infected leaves, follow crop rotation, and use recommended fungicide/insecticide if required."
        }

# (Optional) Override Important diseases with specific data
disease_info.update({
    "Apple___Apple_scab": {
        "symptoms": "Olive-green/dark spots on leaves and fruits; leaves may curl and fall early.",
        "prevention": "Remove infected leaves, prune for airflow, apply copper fungicide early season."
    },
    "Tomato___Late_blight": {
        "symptoms": "Water-soaked lesions, black patches, white mold underside; fruit rot possible.",
        "prevention": "Avoid wet leaves, maintain spacing, apply mancozeb/copper fungicide early."
    },
    "Potato___Late_blight": {
        "symptoms": "Rapid browning of leaves, black edges, tubers may rot.",
        "prevention": "Use healthy seed tubers, avoid moist conditions, apply metalaxyl-M fungicide."
    }
})


# ---------------------- SIDEBAR ---------------------- #
st.sidebar.title("🌿 Plant Guardian")
st.sidebar.markdown(
    "<p style='font-size:16px;'>Upload a leaf image on the Detection tab to identify diseases and get fertilizer advice.</p>",
    unsafe_allow_html=True
)


# ---------------------- TABS ---------------------- #
tab1, tab2 = st.tabs(["🌱 Detection", "📘 Info"])


with tab1:
    st.markdown("## 🌿 Plant Disease Detection")

    # ---------- Session state ----------
    if "source_mode" not in st.session_state:
        st.session_state.source_mode = "Upload"

    if "open_camera" not in st.session_state:
        st.session_state.open_camera = False

    # ---------- Toggle Buttons ----------
    colA, colB = st.columns(2)

    with colA:
        if st.button("📁 Upload", use_container_width=True):
            st.session_state.source_mode = "Upload"
            st.session_state.open_camera = False

    with colB:
        if st.button("📷 Camera", use_container_width=True):
            st.session_state.source_mode = "Camera"
            st.session_state.open_camera = False

    st.markdown(f"<p style='text-align:center; font-size:15px;'>Selected mode: <b>{st.session_state.source_mode}</b></p>",
                unsafe_allow_html=True)

    uploaded_file = None
    camera_file = None

    # ---------- Upload mode ----------
    if st.session_state.source_mode == "Upload":
        col1, col2 = st.columns([4, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose leaf image",
                type=["jpg", "jpeg", "png"],
                label_visibility="visible"
            )

        # Circular camera icon beside uploader
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='circle-btn'>", unsafe_allow_html=True)
            if st.button("📷", help="Switch to camera mode"):
                st.session_state.source_mode = "Camera"
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Camera mode ----------
    if st.session_state.source_mode == "Camera":
        st.markdown("### 📷 Capture Leaf Image")

        # Capture button (camera opens only after click)
        colC1, colC2, colC3 = st.columns([2, 2, 2])

        with colC2:
            st.markdown("<div class='circle-btn'>", unsafe_allow_html=True)
            if st.button("📸", help="Open camera"):
                st.session_state.open_camera = True
            st.markdown("</div>", unsafe_allow_html=True)

        # camera widget shown only after pressing button
        if st.session_state.open_camera:
            st.markdown("<div class='camera-box'>", unsafe_allow_html=True)

            camera_file = st.camera_input("Camera", label_visibility="collapsed")

            st.markdown("</div>", unsafe_allow_html=True)

            # Auto close after capture
            if camera_file is not None:
                st.session_state.open_camera = False

            # Manual close button
            if st.button("❌ Close Camera"):
                st.session_state.open_camera = False

    # ---------- Common handler ----------
    file_source = uploaded_file if uploaded_file is not None else camera_file

    if file_source:
        image = Image.open(file_source).convert("RGB")

        # ✅ continue your prediction code from here...




        # Display uploaded image
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_data = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(
            f"""
            
            /* Source toggle buttons */
.source-btn {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin: 12px 0 18px 0;
}

.source-card {
    padding: 12px 14px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.15);
    cursor: pointer;
    font-weight: 600;
    transition: 0.2s;
    width: 160px;
}

.source-card:hover {
    transform: translateY(-1px);
}

/* Camera popup card */
.camera-box {
    padding: 15px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.12);
    margin-top: 10px;
}

/* Circular icon button */
.circle-btn button {
    border-radius: 999px !important;
    width: 52px !important;
    height: 52px !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    padding: 0px !important;
}

            """, unsafe_allow_html=True
        )

        # Prepare image
        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        pretty_label = prettify_label(predicted_class)

        st.markdown(
            f"<div class='prediction-card'>🔎 <strong>Prediction:</strong> {pretty_label}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='prediction-card'>🎯 <strong>Confidence:</strong> {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )

        # Fertilizer suggestion
        if predicted_class in fertilizer_map:
            tip = fertilizer_map[predicted_class]
            st.markdown(
                f"<div class='prediction-card'>💡 <strong>Fertilizer/Treatment Tip:</strong> {tip}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='prediction-card'>✅ This plant appears healthy. No treatment needed!</div>",
                unsafe_allow_html=True
            )

        # Disease Description
        if predicted_class in disease_info:
            st.markdown("### 🩺 Disease Description")
            st.markdown(
                f"""
                <div class='prediction-card'>
                ✅ <strong>Symptoms:</strong> {disease_info[predicted_class]['symptoms']} <br><br>
                🛡️ <strong>Prevention:</strong> {disease_info[predicted_class]['prevention']}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Grad-CAM Visualization
        st.markdown("### 📊 Grad-CAM: Model Focus Visualization")
        heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name="Conv_1")
        overlay_img = overlay_gradcam(img, heatmap)
        st.image(overlay_img, caption="Grad-CAM: Highlighted Disease Regions", use_container_width=True)


with tab2:
    st.markdown("## 📘 About This App")
    st.markdown("""
    This application helps farmers and gardeners detect plant diseases from leaf images 
    and recommends suitable fertilizers or treatments.

    **Features:**
    - Deep learning–based leaf disease classification  
    - Fertilizer/Treatment recommendations  
    - Symptoms + Prevention guide  
    - Grad-CAM (Explainable AI)  
    - Mobile-friendly responsive layout  
    - Dark mode UI (uses system preference)
    """)


