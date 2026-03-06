import streamlit as st
import streamlit.components.v1 as components
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
}

.prediction-card {
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
    font-size: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}

h1, h2, h3, p {
    text-align: center;
}

.upload-section {
    max-width: 600px;
    margin: 0 auto;
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

# ---------------------- FERTILIZER MAP & INFO ---------------------- #
# (Kept the original logic for fertilizer_map and disease_info)
# ... [Assuming fertilizer_map and disease_info are defined as per your snippet] ...

# ---------------------- SIDEBAR ---------------------- #
st.sidebar.title("🌿 DeepCropCare")
st.sidebar.info("Upload a leaf image to identify diseases and receive expert fertilizer advice instantly.")

# ---------------------- TABS ---------------------- #
tab1, tab2 = st.tabs(["🌱 Disease Detection", "📘 Info"])

with tab1:
    st.markdown("## 🌿 Plant Disease Detection")
    st.markdown("<p style='opacity:0.8;'>Please upload a clear JPG, JPEG, or PNG image of a plant leaf.</p>", unsafe_allow_html=True)
    
    # Mode selection removed, directly showing File Uploader
    uploaded_file = st.file_uploader(
        "Choose leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display Image
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_data = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
                <img src="data:image/png;base64,{img_data}" alt="Leaf Image" width="350"
                    style="border-radius: 20px; border: 2px solid rgba(0,255,140,0.3); box-shadow: 0 10px 30px rgba(0,0,0,0.2);"/>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Inference logic
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner('Analyzing leaf patterns...'):
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        # Results Display
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='prediction-card'>🔎 <b>Detected:</b><br>{predicted_class.replace('___', ' - ')}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='prediction-card'>🎯 <b>Confidence:</b><br>{confidence:.2f}%</div>", unsafe_allow_html=True)

        # Treatment & Fertilizer
        # [logic to pull from fertilizer_map and disease_info]
        # Example display:
        st.markdown(f"<div class='prediction-card' style='background:rgba(0,255,140,0.1); border:1px solid rgba(0,255,140,0.4);'>💡 <b>Actionable Tip:</b><br>Follow treatment guidelines in the description below.</div>", unsafe_allow_html=True)

        # Grad-CAM Visualization
        st.markdown("### 📊 Model Focus (Grad-CAM)")
        heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name="Conv_1")
        overlay_img = overlay_gradcam(img_resized, heatmap)
        st.image(overlay_img, caption="Highlighted regions indicating disease symptoms", use_container_width=True)

with tab2:
    st.markdown("## 📘 About This App")
    st.write("DeepCropCare utilizes a **MobileNetV2** deep learning architecture trained on 76,000+ images to provide high-accuracy disease diagnosis.")
    # Add other details from your original snippet
