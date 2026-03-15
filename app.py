import os
import smtplib
import time
import json
import hashlib
import tempfile
import warnings
from io import BytesIO
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import quote

import cv2
import joblib
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from dotenv import load_dotenv
from fpdf import FPDF
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

try:
    from google import genai as modern_genai
except ImportError:
    modern_genai = None

legacy_genai = None


load_dotenv()

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
}

LANGUAGE_LABELS = {
    "en": {
        "page_title": "DeepCropCare",
        "hero_title": "DeepCropCare",
        "hero_subtitle": "Precision AI for Plant Health & Smarter Yields",
        "hero_badge": "Leaf intelligence for modern farming",
        "hero_description": "Diagnose plant stress, recommend the next crop, and ask follow-up questions inside one calm, precise workspace.",
        "language_selector": "Language",
        "tab_disease": "Disease Detection",
        "tab_crop": "Crop Recommendation",
        "tab_chat": "Agronomist AI",
        "tab_info": "Project Info",
        "choose_workspace": "Choose a workspace",
        "launcher_note": "Tap an icon to switch instantly between tools.",
        "feature_disease_subtitle": "Upload a leaf image, run the model, inspect the confidence and Grad-CAM heatmap, then export a polished report.",
        "feature_crop_subtitle": "Blend soil nutrients, rainfall, humidity, and temperature into a clearer crop recommendation with practical guidance.",
        "feature_chat_subtitle": "Ask follow-up questions, simplify agronomy advice, and keep the conversation connected to the latest diagnosis.",
        "feature_info_subtitle": "A quick look at the computer vision stack, crop recommendation pipeline, and interpretability layer behind the app.",
        "disease_heading": "Plant Disease Analysis",
        "upload_mode_label": "Choose input source",
        "browse_files": "Browse Files",
        "camera": "Camera",
        "capture_leaf": "Capture leaf image",
        "upload_leaf": "Upload leaf image",
        "uploaded_specimen": "Uploaded Specimen",
        "run_analysis": "Run Diagnostic Analysis",
        "identifying": "Identifying pathogens...",
        "analysis_complete": "Analysis Complete!",
        "confidence": "Confidence",
        "recommended_action": "Fertilizer Advisory",
        "report_summary": "Diagnostic Report Summary",
        "report_download": "Download Report",
        "report_email": "Email Report",
        "report_subject": "DeepCropCare Disease Report",
        "report_generated_on": "Generated On",
        "report_image_name": "Image Name",
        "report_detected_disease": "Detected Disease",
        "report_heatmap_status": "Heatmap Analysis",
        "report_heatmap_ready": "Included",
        "report_heatmap_unavailable": "Not included",
        "report_input_summary": "Input Summary",
        "report_weather_used": "Weather Used",
        "report_email_body_intro": "Please find the report details below.",
        "crop_report_subject": "DeepCropCare Crop Recommendation Report",
        "crop_report_heading": "Crop Recommendation Report",
        "crop_report_download": "Download Crop Report",
        "crop_report_email": "Email Crop Report",
        "report_temperature": "Temperature",
        "report_crop_name": "Recommended Crop",
        "report_crop_description": "Crop Description",
        "report_crop_conditions": "Optimal Conditions",
        "report_crop_fertilizer": "Fertilizer and Care",
        "report_crop_tip": "Pro Tip",
        "email_address": "Recipient Email",
        "send_email": "Send Email",
        "email_sent": "Email sent successfully.",
        "email_failed": "Unable to send email.",
        "email_config_missing": "Email service is not configured. Add SMTP settings in Streamlit secrets or environment variables.",
        "email_required": "Enter a valid email address.",
        "email_report_heading": "Send Report by Email",
        "heatmap_title": "AI Heatmap: Detected Infection Zones",
        "original_scan": "Original Scan",
        "infection_hotspots": "Infection Hotspots",
        "visualization_error": "Visualization error",
        "disease_model_missing": "Disease model not loaded.",
        "crop_heading": "Smart Crop Recommendation",
        "soil_parameters": "Soil Parameters",
        "nitrogen": "Nitrogen (N)",
        "phosphorus": "Phosphorus (P)",
        "potassium": "Potassium (K)",
        "soil_ph": "Soil pH Level",
        "rainfall": "Annual Rainfall (mm)",
        "weather_heading": "Local Weather",
        "city_input": "Enter Village & District",
        "fetch_weather": "Fetch Live Weather",
        "weather_success": "Data fetched for",
        "weather_error": "Village not found. Try 'District, State'",
        "temp": "Temp (°C)",
        "humidity": "Humidity (%)",
        "recommend_crop": "Recommend Best Crop",
        "recommended_crop": "Recommended",
        "crop_model_missing": "Crop model not loaded! Defaulting to demo mode.",
        "description": "Description",
        "optimal_conditions": "Optimal Conditions",
        "fertilizer_care": "Fertilizer & Care Advice",
        "pro_tip": "Pro-Tip",
        "crop_analysis_complete": "Analysis Complete",
        "chat_heading": "DeepCropCare Agronomist AI",
        "api_missing": "API Key Missing: Please add it to Streamlit Secrets.",
        "chat_welcome": "Hi! I am your Agronomist AI. How can I help you today?",
        "chat_placeholder": "Ask about fertilizers, pests, or soil...",
        "chat_spinner": "Consulting Agronomist...",
        "quota_error": "Daily quota exhausted. Try again tomorrow.",
        "generic_error": "Error",
        "reset_chat": "Reset Chat",
        "system_instruction": "You are a professional Agronomist AI. The user's plant has {disease}. Reply in English. Be concise, use bullet points, and provide expert farming advice.",
        "assistant_hint": "Tap the helper bot for more details.",
        "assistant_note": "Agronomist helper",
        "assistant_trigger": "Open agronomist helper",
        "assistant_auto_prompt": "Explain the detected disease {disease} in simple words. Give a detailed but easy-to-understand answer with these sections: 1. What this disease is. 2. Main symptoms farmers can notice. 3. Cure or treatment steps to follow now. 4. Prevention tips for the next few days and next season. 5. Fertilizer and plant care advice. 6. Simple do and don't points. Keep the tone practical and farmer-friendly.",
        "assistant_fallback_title": "Quick Agronomist Guide",
        "assistant_fallback_description": "This looks like {disease}. It can stress the plant and reduce growth if not managed early.",
        "assistant_fallback_symptoms": "Check for spots, color changes, wilting, damaged leaf areas, or poor plant vigor.",
        "assistant_fallback_cure": "Remove badly affected leaves, improve airflow, avoid leaf wetness for long periods, and follow the recommended treatment below.",
        "assistant_fallback_prevention": "Use clean tools, avoid overcrowding, monitor plants daily, and act early when symptoms first appear.",
        "assistant_fallback_fertilizer": "Recommended care: {advice}",
        "label_symptoms": "Symptoms",
        "label_cure": "Cure",
        "label_prevention": "Prevention",
        "label_fertilizer_care": "Fertilizer & Care",
        "heatmap_images_title": "AI Heatmap Images",
        "about_heading": "About DeepCropCare",
        "mission_title": "The Mission",
        "mission_body": "DeepCropCare is a cutting-edge agricultural platform designed to empower farmers with data-driven insights. By merging Deep Learning for leaf diagnostics and Machine Learning for crop suitability, we provide a 360-degree view of your farm's potential and health.",
        "cv_title": "Computer Vision",
        "cv_body": "Using Convolutional Neural Networks (CNN), the system analyzes leaf texture and patterns to detect multiple plant disease states.",
        "ml_title": "Predictive Analytics",
        "ml_body": "Our Random Forest engine processes soil NPK levels and weather data to recommend the ideal crop for your land.",
        "npk_title": "Understanding Soil Parameters (N-P-K)",
        "npk_body": [
            "Nitrogen (N): Essential for leaf growth and green color.",
            "Phosphorus (P): Critical for root development and flower/seed production.",
            "Potassium (K): Helps with overall plant health and disease resistance.",
        ],
        "interpretability": "Interpretability: How the AI Sees",
        "target_layer": "Target Diagnostic Layer",
        "target_layer_desc": "The heatmap highlights areas the AI identified as diseased.",
        "footer": "DeepCropCare v1.0 | 2026 Agricultural Innovation",
        "general_farming": "general farming",
        "na": "N/A",
        "weather_unavailable": "Weather service unavailable",
        "healthy_word": "healthy",
    },
    "hi": {
        "page_title": "डीपक्रॉपकेयर",
        "hero_title": "डीपक्रॉपकेयर",
        "hero_subtitle": "पौधों के स्वास्थ्य और बेहतर उपज के लिए सटीक एआई",
        "hero_badge": "आधुनिक खेती के लिए पत्ती इंटेलिजेंस",
        "hero_description": "पौधों पर तनाव पहचानें, अगली फसल सुझाएँ और एक शांत, सटीक कार्यक्षेत्र में आगे के प्रश्न पूछें।",
        "language_selector": "भाषा",
        "tab_disease": "रोग पहचान",
        "tab_crop": "फसल सिफारिश",
        "tab_chat": "कृषि विशेषज्ञ एआई",
        "tab_info": "परियोजना जानकारी",
        "choose_workspace": "कार्यक्षेत्र चुनें",
        "launcher_note": "उपकरण बदलने के लिए किसी आइकन पर टैप करें।",
        "feature_disease_subtitle": "पत्ती की छवि अपलोड करें, मॉडल चलाएँ, विश्वास स्तर और Grad-CAM हीटमैप देखें, फिर एक साफ रिपोर्ट निर्यात करें।",
        "feature_crop_subtitle": "मिट्टी के पोषक तत्व, वर्षा, आर्द्रता और तापमान को जोड़कर अधिक स्पष्ट फसल सिफारिश प्राप्त करें।",
        "feature_chat_subtitle": "आगे के प्रश्न पूछें, कृषि सलाह को सरल बनाएं और बातचीत को नवीनतम निदान से जोड़े रखें।",
        "feature_info_subtitle": "कंप्यूटर विज़न, फसल सिफारिश पाइपलाइन और व्याख्यात्मक एआई पर एक त्वरित नज़र।",
        "disease_heading": "पौध रोग विश्लेषण",
        "upload_mode_label": "इनपुट स्रोत चुनें",
        "browse_files": "फ़ाइल चुनें",
        "camera": "कैमरा",
        "capture_leaf": "पत्ती की तस्वीर लें",
        "upload_leaf": "पत्ती की छवि अपलोड करें",
        "uploaded_specimen": "अपलोड किया गया नमूना",
        "run_analysis": "डायग्नोस्टिक विश्लेषण चलाएँ",
        "identifying": "रोगजनकों की पहचान की जा रही है...",
        "analysis_complete": "विश्लेषण पूरा हुआ!",
        "confidence": "विश्वास स्तर",
        "recommended_action": "उर्वरक सलाह",
        "report_summary": "डायग्नोस्टिक रिपोर्ट सारांश",
        "report_download": "रिपोर्ट डाउनलोड करें",
        "report_email": "रिपोर्ट ईमेल करें",
        "report_subject": "डीपक्रॉपकेयर रोग रिपोर्ट",
        "report_generated_on": "रिपोर्ट तिथि",
        "report_image_name": "छवि का नाम",
        "report_detected_disease": "पहचाना गया रोग",
        "report_heatmap_status": "हीटमैप विश्लेषण",
        "report_heatmap_ready": "शामिल है",
        "report_heatmap_unavailable": "शामिल नहीं है",
        "report_input_summary": "इनपुट सारांश",
        "report_weather_used": "उपयोग किया गया मौसम",
        "report_email_body_intro": "कृपया नीचे रिपोर्ट का विवरण देखें।",
        "crop_report_subject": "डीपक्रॉपकेयर फसल सिफारिश रिपोर्ट",
        "crop_report_heading": "फसल सिफारिश रिपोर्ट",
        "crop_report_download": "फसल रिपोर्ट डाउनलोड करें",
        "crop_report_email": "फसल रिपोर्ट ईमेल करें",
        "report_temperature": "तापमान",
        "report_crop_name": "अनुशंसित फसल",
        "report_crop_description": "फसल विवरण",
        "report_crop_conditions": "उत्तम परिस्थितियाँ",
        "report_crop_fertilizer": "उर्वरक और देखभाल",
        "report_crop_tip": "विशेष सुझाव",
        "email_address": "प्राप्तकर्ता ईमेल",
        "send_email": "ईमेल भेजें",
        "email_sent": "ईमेल सफलतापूर्वक भेज दिया गया।",
        "email_failed": "ईमेल भेजा नहीं जा सका।",
        "email_config_missing": "ईमेल सेवा कॉन्फ़िगर नहीं है। Streamlit secrets या environment variables में SMTP सेटिंग्स जोड़ें।",
        "email_required": "कृपया सही ईमेल पता दर्ज करें।",
        "email_report_heading": "ईमेल द्वारा रिपोर्ट भेजें",
        "heatmap_title": "एआई हीटमैप: पहचाने गए संक्रमण क्षेत्र",
        "original_scan": "मूल स्कैन",
        "infection_hotspots": "संक्रमण हॉटस्पॉट",
        "visualization_error": "विज़ुअलाइज़ेशन त्रुटि",
        "disease_model_missing": "रोग मॉडल लोड नहीं हुआ।",
        "crop_heading": "स्मार्ट फसल सिफारिश",
        "soil_parameters": "मिट्टी के पैरामीटर",
        "nitrogen": "नाइट्रोजन (N)",
        "phosphorus": "फॉस्फोरस (P)",
        "potassium": "पोटैशियम (K)",
        "soil_ph": "मिट्टी का pH स्तर",
        "rainfall": "वार्षिक वर्षा (मिमी)",
        "weather_heading": "स्थानीय मौसम",
        "city_input": "गांव और जिला दर्ज करें",
        "fetch_weather": "लाइव मौसम लाएँ",
        "weather_success": "डेटा प्राप्त किया गया:",
        "weather_error": "गांव नहीं मिला। 'जिला, राज्य' आज़माएँ",
        "temp": "तापमान (°C)",
        "humidity": "आर्द्रता (%)",
        "recommend_crop": "सर्वश्रेष्ठ फसल सुझाएँ",
        "recommended_crop": "अनुशंसित",
        "crop_model_missing": "फसल मॉडल लोड नहीं हुआ। डेमो मोड उपयोग किया जा रहा है।",
        "description": "विवरण",
        "optimal_conditions": "उत्तम परिस्थितियाँ",
        "fertilizer_care": "उर्वरक और देखभाल सलाह",
        "pro_tip": "विशेष सुझाव",
        "crop_analysis_complete": "विश्लेषण पूरा हुआ",
        "chat_heading": "डीपक्रॉपकेयर कृषि विशेषज्ञ एआई",
        "api_missing": "API कुंजी उपलब्ध नहीं है: कृपया इसे Streamlit Secrets में जोड़ें।",
        "chat_welcome": "नमस्ते! मैं आपका कृषि विशेषज्ञ एआई हूँ। आज मैं आपकी कैसे मदद कर सकता हूँ?",
        "chat_placeholder": "उर्वरक, कीट या मिट्टी के बारे में पूछें...",
        "chat_spinner": "कृषि विशेषज्ञ से परामर्श लिया जा रहा है...",
        "quota_error": "आज की सीमा समाप्त हो गई है। कल फिर प्रयास करें।",
        "generic_error": "त्रुटि",
        "reset_chat": "चैट रीसेट करें",
        "system_instruction": "आप एक पेशेवर कृषि विशेषज्ञ एआई हैं। उपयोगकर्ता के पौधे में {disease} है। केवल हिंदी में उत्तर दें। संक्षिप्त रहें, बुलेट पॉइंट्स का उपयोग करें और विशेषज्ञ कृषि सलाह दें।",
        "assistant_hint": "अधिक जानकारी के लिए सहायक बॉट पर टैप करें।",
        "assistant_note": "कृषि सहायक",
        "assistant_trigger": "कृषि सहायक खोलें",
        "assistant_auto_prompt": "पता चली हुई बीमारी {disease} को बहुत आसान हिंदी में समझाइए। उत्तर को इन भागों में दीजिए: 1. यह बीमारी क्या है। 2. किसान कौन से मुख्य लक्षण देख सकते हैं। 3. अभी कौन सा उपचार करना चाहिए। 4. अगले कुछ दिनों और अगले सीजन के लिए बचाव के तरीके। 5. उर्वरक और पौध देखभाल सलाह। 6. आसान क्या करें और क्या न करें। उत्तर व्यावहारिक और किसान-मित्र होना चाहिए।",
        "assistant_fallback_title": "त्वरित कृषि मार्गदर्शिका",
        "assistant_fallback_description": "यह {disease} जैसा दिख रहा है। समय पर नियंत्रण न करने पर पौधे की वृद्धि और उत्पादन प्रभावित हो सकते हैं।",
        "assistant_fallback_symptoms": "धब्बे, रंग बदलना, मुरझाना, पत्ती का खराब होना या पौधे की कमजोरी जैसे लक्षण देखें।",
        "assistant_fallback_cure": "बहुत प्रभावित पत्तियाँ हटाएँ, हवा का प्रवाह बढ़ाएँ, पत्तियों पर लंबे समय तक नमी न रहने दें और नीचे दी गई सलाह का पालन करें।",
        "assistant_fallback_prevention": "साफ औज़ार रखें, पौधों को बहुत घना न लगाएँ, रोज़ निगरानी करें और शुरुआती लक्षण पर तुरंत कार्रवाई करें।",
        "assistant_fallback_fertilizer": "अनुशंसित देखभाल: {advice}",
        "label_symptoms": "लक्षण",
        "label_cure": "उपचार",
        "label_prevention": "रोकथाम",
        "label_fertilizer_care": "उर्वरक और देखभाल",
        "heatmap_images_title": "एआई हीटमैप चित्र",
        "about_heading": "डीपक्रॉपकेयर के बारे में",
        "mission_title": "हमारा मिशन",
        "mission_body": "डीपक्रॉपकेयर एक उन्नत कृषि मंच है जो किसानों को डेटा-आधारित जानकारी देकर सशक्त बनाता है। पत्ती रोग पहचान के लिए डीप लर्निंग और फसल उपयुक्तता के लिए मशीन लर्निंग को जोड़कर यह आपकी खेती की क्षमता और स्वास्थ्य का समग्र दृश्य देता है।",
        "cv_title": "कंप्यूटर विज़न",
        "cv_body": "कन्वोल्यूशनल न्यूरल नेटवर्क (CNN) का उपयोग करके यह प्रणाली पत्तियों की बनावट और पैटर्न का विश्लेषण करके कई पौध रोग अवस्थाओं का पता लगाती है।",
        "ml_title": "पूर्वानुमान विश्लेषण",
        "ml_body": "हमारा रैंडम फॉरेस्ट इंजन मिट्टी के NPK स्तर और मौसम डेटा को संसाधित करके आपकी भूमि के लिए उपयुक्त फसल सुझाता है।",
        "npk_title": "मिट्टी के पैरामीटर (N-P-K) को समझें",
        "npk_body": [
            "नाइट्रोजन (N): पत्तियों की वृद्धि और हरे रंग के लिए आवश्यक।",
            "फॉस्फोरस (P): जड़ों के विकास और फूल/बीज उत्पादन के लिए महत्वपूर्ण।",
            "पोटैशियम (K): पौधे के समग्र स्वास्थ्य और रोग प्रतिरोध में सहायक।",
        ],
        "interpretability": "व्याख्यात्मकता: एआई कैसे देखता है",
        "target_layer": "लक्षित डायग्नोस्टिक लेयर",
        "target_layer_desc": "हीटमैप उन क्षेत्रों को दिखाता है जिन्हें एआई ने रोगग्रस्त पहचाना।",
        "footer": "डीपक्रॉपकेयर v1.0 | 2026 कृषि नवाचार",
        "general_farming": "सामान्य खेती",
        "na": "लागू नहीं",
        "weather_unavailable": "मौसम सेवा उपलब्ध नहीं है",
        "healthy_word": "स्वस्थ",
    },
    "te": {
        "page_title": "డీప్‌క్రాప్‌కేర్",
        "hero_title": "డీప్‌క్రాప్‌కేర్",
        "hero_subtitle": "మొక్కల ఆరోగ్యం మరియు మెరుగైన దిగుబడికి ఖచ్చితమైన ఏఐ",
        "hero_badge": "ఆధునిక వ్యవసాయానికి ఆకు ఇంటెలిజెన్స్",
        "hero_description": "మొక్కల ఒత్తిడిని గుర్తించండి, తదుపరి పంటను సూచించండి మరియు ఒక ప్రశాంతమైన, ఖచ్చితమైన వర్క్‌స్పేస్‌లో ఫాలో-అప్ ప్రశ్నలు అడగండి.",
        "language_selector": "భాష",
        "tab_disease": "వ్యాధి గుర్తింపు",
        "tab_crop": "పంట సిఫార్సు",
        "tab_chat": "వ్యవసాయ నిపుణుడు ఏఐ",
        "tab_info": "ప్రాజెక్ట్ సమాచారం",
        "choose_workspace": "వర్క్‌స్పేస్ ఎంచుకోండి",
        "launcher_note": "పరికరాన్ని వెంటనే మార్చడానికి ఐకాన్‌పై ట్యాప్ చేయండి.",
        "feature_disease_subtitle": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేసి, మోడల్‌ను నడపండి, విశ్వాస స్థాయి మరియు Grad-CAM హీట్‌మ్యాప్‌ను పరిశీలించి, రిపోర్ట్‌ను ఎగుమతి చేయండి.",
        "feature_crop_subtitle": "మట్టిపోషకాలు, వర్షపాతం, ఆర్ద్రత మరియు ఉష్ణోగ్రత ఆధారంగా మరింత స్పష్టమైన పంట సిఫార్సు పొందండి.",
        "feature_chat_subtitle": "ఫాలో-అప్ ప్రశ్నలు అడిగి, వ్యవసాయ సలహాను సరళీకరించి, తాజా నిర్ధారణకు సంభాషణను అనుసంధానంగా ఉంచండి.",
        "feature_info_subtitle": "కంప్యూటర్ విజన్, పంట సిఫార్సు పైప్‌లైన్ మరియు ఇంటర్‌ప్రిటబిలిటీ లేయర్‌పై ఒక త్వరిత దర్శనం.",
        "disease_heading": "మొక్కల వ్యాధి విశ్లేషణ",
        "upload_mode_label": "ఇన్‌పుట్ మూలాన్ని ఎంచుకోండి",
        "browse_files": "ఫైళ్లు ఎంచుకోండి",
        "camera": "కెమెరా",
        "capture_leaf": "ఆకు ఫోటో తీయండి",
        "upload_leaf": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "uploaded_specimen": "అప్‌లోడ్ చేసిన నమూనా",
        "run_analysis": "నిర్ధారణ విశ్లేషణ ప్రారంభించండి",
        "identifying": "రోగ కారకాలను గుర్తిస్తోంది...",
        "analysis_complete": "విశ్లేషణ పూర్తైంది!",
        "confidence": "నమ్మక స్థాయి",
        "recommended_action": "ఎరువు సలహా",
        "report_summary": "డయాగ్నస్టిక్ రిపోర్ట్ సారాంశం",
        "report_download": "రిపోర్ట్ డౌన్‌లోడ్ చేయండి",
        "report_email": "రిపోర్ట్ ఈమెయిల్ చేయండి",
        "report_subject": "డీప్‌క్రాప్‌కేర్ వ్యాధి రిపోర్ట్",
        "report_generated_on": "రిపోర్ట్ తయారైన సమయం",
        "report_image_name": "చిత్రం పేరు",
        "report_detected_disease": "గుర్తించిన వ్యాధి",
        "report_heatmap_status": "హీట్‌మ్యాప్ విశ్లేషణ",
        "report_heatmap_ready": "చేర్చబడింది",
        "report_heatmap_unavailable": "చేర్చబడలేదు",
        "report_input_summary": "ఇన్‌పుట్ సారాంశం",
        "report_weather_used": "ఉపయోగించిన వాతావరణం",
        "report_email_body_intro": "దయచేసి క్రింది రిపోర్ట్ వివరాలను చూడండి.",
        "crop_report_subject": "డీప్‌క్రాప్‌కేర్ పంట సిఫార్సు రిపోర్ట్",
        "crop_report_heading": "పంట సిఫార్సు రిపోర్ట్",
        "crop_report_download": "పంట రిపోర్ట్ డౌన్‌లోడ్ చేయండి",
        "crop_report_email": "పంట రిపోర్ట్ ఈమెయిల్ చేయండి",
        "report_temperature": "ఉష్ణోగ్రత",
        "report_crop_name": "సిఫార్సు చేసిన పంట",
        "report_crop_description": "పంట వివరణ",
        "report_crop_conditions": "అనుకూల పరిస్థితులు",
        "report_crop_fertilizer": "ఎరువు మరియు సంరక్షణ",
        "report_crop_tip": "ప్రో చిట్కా",
        "email_address": "స్వీకర్త ఈమెయిల్",
        "send_email": "ఈమెయిల్ పంపండి",
        "email_sent": "ఈమెయిల్ విజయవంతంగా పంపబడింది.",
        "email_failed": "ఈమెయిల్ పంపలేకపోయాం.",
        "email_config_missing": "ఈమెయిల్ సేవ సెట్ కాలేదు. Streamlit secrets లేదా environment variables లో SMTP సెట్టింగ్స్ జోడించండి.",
        "email_required": "సరైన ఈమెయిల్ చిరునామా ఇవ్వండి.",
        "email_report_heading": "ఈమెయిల్ ద్వారా రిపోర్ట్ పంపండి",
        "heatmap_title": "ఏఐ హీట్‌మ్యాప్: గుర్తించిన సంక్రమణ ప్రాంతాలు",
        "original_scan": "మూల స్కాన్",
        "infection_hotspots": "సంక్రమణ హాట్‌స్పాట్లు",
        "visualization_error": "విజువలైజేషన్ లోపం",
        "disease_model_missing": "వ్యాధి మోడల్ లోడ్ కాలేదు.",
        "crop_heading": "స్మార్ట్ పంట సిఫార్సు",
        "soil_parameters": "మట్టిపరామితులు",
        "nitrogen": "నైట్రోజన్ (N)",
        "phosphorus": "ఫాస్ఫరస్ (P)",
        "potassium": "పొటాషియం (K)",
        "soil_ph": "మట్టి pH స్థాయి",
        "rainfall": "వార్షిక వర్షపాతం (మి.మీ.)",
        "weather_heading": "స్థానిక వాతావరణం",
        "city_input": "గ్రామం మరియు జిల్లా నమోదు చేయండి",
        "fetch_weather": "ప్రత్యక్ష వాతావరణం తెచ్చుకోండి",
        "weather_success": "డేటా తెచ్చుకున్నాం:",
        "weather_error": "గ్రామం కనబడలేదు. 'జిల్లా, రాష్ట్రం' ప్రయత్నించండి",
        "temp": "ఉష్ణోగ్రత (°C)",
        "humidity": "ఆర్ద్రత (%)",
        "recommend_crop": "ఉత్తమ పంటను సూచించండి",
        "recommended_crop": "సిఫార్సు చేయబడినది",
        "crop_model_missing": "పంట మోడల్ లోడ్ కాలేదు. డెమో మోడ్ ఉపయోగిస్తున్నాం.",
        "description": "వివరణ",
        "optimal_conditions": "అనుకూల పరిస్థితులు",
        "fertilizer_care": "ఎరువు మరియు సంరక్షణ సలహా",
        "pro_tip": "ప్రో చిట్కా",
        "crop_analysis_complete": "విశ్లేషణ పూర్తైంది",
        "chat_heading": "డీప్‌క్రాప్‌కేర్ వ్యవసాయ నిపుణుడు ఏఐ",
        "api_missing": "API కీ లేదు: దయచేసి Streamlit Secrets లో జోడించండి.",
        "chat_welcome": "హాయ్! నేను మీ వ్యవసాయ నిపుణుడు ఏఐని. ఈరోజు నేను ఎలా సహాయం చేయగలను?",
        "chat_placeholder": "ఎరువులు, పురుగులు లేదా మట్టిపై అడగండి...",
        "chat_spinner": "వ్యవసాయ నిపుణుడిని సంప్రదిస్తోంది...",
        "quota_error": "రోజువారీ పరిమితి పూర్తైంది. రేపు మళ్లీ ప్రయత్నించండి.",
        "generic_error": "లోపం",
        "reset_chat": "చాట్ రీసెట్ చేయండి",
        "system_instruction": "మీరు ఒక ప్రొఫెషనల్ వ్యవసాయ నిపుణుడు ఏఐ. వినియోగదారుడి మొక్కకు {disease} ఉంది. తెలుగు లో మాత్రమే జవాబివ్వండి. సంక్షిప్తంగా, బుల్లెట్ పాయింట్లలో నిపుణుల వ్యవసాయ సలహా ఇవ్వండి.",
        "assistant_hint": "మరిన్ని వివరాల కోసం సహాయక బాట్‌పై ట్యాప్ చేయండి.",
        "assistant_note": "వ్యవసాయ సహాయకుడు",
        "assistant_trigger": "వ్యవసాయ సహాయకుడిని తెరవండి",
        "assistant_auto_prompt": "గుర్తించిన వ్యాధి {disease} ను చాలా సులభమైన తెలుగులో వివరించండి. జవాబును ఈ భాగాలుగా ఇవ్వండి: 1. ఈ వ్యాధి ఏమిటి. 2. రైతు గమనించే ప్రధాన లక్షణాలు. 3. ఇప్పుడు చేయాల్సిన చికిత్స. 4. వచ్చే కొన్ని రోజులు మరియు వచ్చే సీజన్‌కు నివారణ సూచనలు. 5. ఎరువు మరియు మొక్క సంరక్షణ సలహా. 6. సులభమైన చేయాల్సినవి, చేయకూడనివి. జవాబు రైతులకు సులభంగా అర్థమయ్యేలా ఉండాలి.",
        "assistant_fallback_title": "త్వరిత వ్యవసాయ మార్గదర్శి",
        "assistant_fallback_description": "ఇది {disease} లాగా కనిపిస్తోంది. తొందరగా నియంత్రించకపోతే మొక్క పెరుగుదల మరియు దిగుబడిపై ప్రభావం పడుతుంది.",
        "assistant_fallback_symptoms": "మచ్చలు, రంగు మార్పు, వాడిపోవడం, ఆకు దెబ్బతినడం లేదా మొక్క బలహీనత వంటి లక్షణాలు చూడండి.",
        "assistant_fallback_cure": "బాగా సోకిన ఆకులు తొలగించండి, గాలి సరిగా ఆడేలా చూడండి, ఆకులపై ఎక్కువసేపు తేమ ఉండనివ్వకండి మరియు క్రింది సలహాను పాటించండి.",
        "assistant_fallback_prevention": "పరికరాలు శుభ్రంగా ఉంచండి, మొక్కలు గట్టిగా నింపవద్దు, రోజూ గమనించండి, మొదటి లక్షణాలకే చర్య తీసుకోండి.",
        "assistant_fallback_fertilizer": "సిఫార్సు చేసిన సంరక్షణ: {advice}",
        "label_symptoms": "లక్షణాలు",
        "label_cure": "చికిత్స",
        "label_prevention": "నివారణ",
        "label_fertilizer_care": "ఎరువు మరియు సంరక్షణ",
        "heatmap_images_title": "ఏఐ హీట్‌మ్యాప్ చిత్రాలు",
        "about_heading": "డీప్‌క్రాప్‌కేర్ గురించి",
        "mission_title": "మా లక్ష్యం",
        "mission_body": "డీప్‌క్రాప్‌కేర్ రైతులకు డేటా ఆధారిత అవగాహనను అందించే ఆధునిక వ్యవసాయ వేదిక. ఆకు వ్యాధి నిర్ధారణకు డీప్ లెర్నింగ్ మరియు పంట అనుకూలతకు మెషీన్ లెర్నింగ్‌ను కలిపి మీ పొలం సామర్థ్యం మరియు ఆరోగ్యంపై సమగ్ర దృశ్యాన్ని అందిస్తుంది.",
        "cv_title": "కంప్యూటర్ విజన్",
        "cv_body": "కన్వల్యూషనల్ న్యూరల్ నెట్‌వర్క్స్ (CNN) సహాయంతో ఈ వ్యవస్థ ఆకుల ఆకృతి మరియు నమూనాలను విశ్లేషించి అనేక మొక్కల వ్యాధి స్థితులను గుర్తిస్తుంది.",
        "ml_title": "ప్రిడిక్టివ్ అనలిటిక్స్",
        "ml_body": "మా ర్యాండమ్ ఫారెస్ట్ ఇంజిన్ మట్టి NPK స్థాయిలు మరియు వాతావరణ డేటాను విశ్లేషించి మీ భూమికి సరైన పంటను సూచిస్తుంది.",
        "npk_title": "మట్టి పరామితులు (N-P-K) అర్థం చేసుకోండి",
        "npk_body": [
            "నైట్రోజన్ (N): ఆకుల వృద్ధి మరియు ఆకుపచ్చ రంగుకు అవసరం.",
            "ఫాస్ఫరస్ (P): వేర్ల అభివృద్ధి మరియు పుష్పం/విత్తన ఉత్పత్తికి ముఖ్యమైనది.",
            "పొటాషియం (K): మొక్కల మొత్తం ఆరోగ్యం మరియు వ్యాధి నిరోధకతకు సహాయపడుతుంది.",
        ],
        "interpretability": "వ్యాఖ్యాన సామర్థ్యం: ఏఐ ఎలా చూస్తుంది",
        "target_layer": "లక్ష్య నిర్ధారణ లేయర్",
        "target_layer_desc": "హీట్‌మ్యాప్‌లో ఏఐ వ్యాధిగ్రస్తంగా గుర్తించిన ప్రాంతాలు చూపబడతాయి.",
        "footer": "డీప్‌క్రాప్‌కేర్ v1.0 | 2026 వ్యవసాయ వినూత్నత",
        "general_farming": "సాధారణ వ్యవసాయం",
        "na": "వర్తించదు",
        "weather_unavailable": "వాతావరణ సేవ అందుబాటులో లేదు",
        "healthy_word": "ఆరోగ్యకరమైనది",
    },
}

DISEASE_METADATA = {
    "Apple___Apple_scab": {
        "display": {"en": "Apple Apple Scab", "hi": "सेब का स्कैब", "te": "ఆపిల్ స్క్యాబ్"},
        "advice": {
            "en": "Use copper-based fungicides (Liquid Copper).",
            "hi": "तांबा-आधारित फफूंदनाशक (लिक्विड कॉपर) का उपयोग करें।",
            "te": "రాగి ఆధారిత ఫంగిసైడ్లు (లిక్విడ్ కాపర్) ఉపయోగించండి.",
        },
    },
    "Apple___Black_rot": {
        "display": {"en": "Apple Black Rot", "hi": "सेब का ब्लैक रॉट", "te": "ఆపిల్ బ్లాక్ రాట్"},
        "advice": {
            "en": "Apply sulfur sprays or captan and prune cankers.",
            "hi": "सल्फर स्प्रे या कैप्टान का प्रयोग करें और कैंकर वाले भाग छाँटें।",
            "te": "సల్ఫర్ స్ప్రే లేదా క్యాప్టాన్ వాడి, కాంకర్లు ఉన్న భాగాలను కత్తిరించండి.",
        },
    },
    "Apple___Cedar_apple_rust": {
        "display": {"en": "Apple Cedar Rust", "hi": "सेब का सीडर रस्ट", "te": "ఆపిల్ సీడర్ రస్ట్"},
        "advice": {
            "en": "Use myclobutanil or mancozeb during spring.",
            "hi": "वसंत ऋतु में मायक्लोब्यूटानिल या मैनकोजेब का उपयोग करें।",
            "te": "వసంతకాలంలో మైక్లోబ్యూటానిల్ లేదా మ్యాంకోజెబ్ ఉపయోగించండి.",
        },
    },
    "Apple___healthy": {
        "display": {"en": "Apple Healthy", "hi": "स्वस्थ सेब", "te": "ఆరోగ్యకరమైన ఆపిల్"},
        "advice": {
            "en": "Maintain soil organic matter with compost.",
            "hi": "कम्पोस्ट से मिट्टी में जैविक पदार्थ बनाए रखें।",
            "te": "కాంపోస్ట్‌తో మట్టిలో సేంద్రీయ పదార్థాన్ని నిల్వ ఉంచండి.",
        },
    },
    "Background_without_leaves": {
        "display": {"en": "Background Without Leaves", "hi": "पत्तियों के बिना पृष्ठभूमि", "te": "ఆకులు లేని నేపథ్యం"},
        "advice": {"en": "N/A", "hi": "लागू नहीं", "te": "వర్తించదు"},
    },
    "Bitter Gourd__Downy_mildew": {
        "display": {"en": "Bitter Gourd Downy Mildew", "hi": "करेले का डाउनी मिल्ड्यू", "te": "కాకరకాయ డౌనీ మిల్డ్యూ"},
        "advice": {
            "en": "Apply Mancozeb or Copper Oxychloride spray.",
            "hi": "मैनकोजेब या कॉपर ऑक्सीक्लोराइड स्प्रे करें।",
            "te": "మ్యాంకోజెబ్ లేదా కాపర్ ఆక్సీక్లోరైడ్ స్ప్రే చేయండి.",
        },
    },
    "Bitter Gourd__Fusarium_wilt": {
        "display": {"en": "Bitter Gourd Fusarium Wilt", "hi": "करेले का फ्यूजेरियम विल्ट", "te": "కాకరకాయ ఫ్యూసేరియం విల్ట్"},
        "advice": {
            "en": "Soil drenching with Carbendazim (0.1%).",
            "hi": "कार्बेन्डाजिम (0.1%) से मिट्टी में ड्रेंचिंग करें।",
            "te": "కార్బెండాజిం (0.1%) తో మట్టికి డ్రెంచింగ్ చేయండి.",
        },
    },
    "Bitter Gourd__Fresh_leaf": {
        "display": {"en": "Bitter Gourd Fresh Leaf", "hi": "करेले की ताज़ी पत्ती", "te": "తాజా కాకరకాయ ఆకు"},
        "advice": {
            "en": "Apply balanced NPK (10-10-10).",
            "hi": "संतुलित NPK (10-10-10) दें।",
            "te": "సంతులిత NPK (10-10-10) ఎరువు ఇవ్వండి.",
        },
    },
    "Bitter Gourd__Mosaic_virus": {
        "display": {"en": "Bitter Gourd Mosaic Virus", "hi": "करेले का मोज़ेक वायरस", "te": "కాకరకాయ మోసాయిక్ వైరస్"},
        "advice": {
            "en": "Control aphids with neem oil and remove infected vines.",
            "hi": "नीम तेल से एफिड नियंत्रित करें और संक्रमित बेलें हटाएँ।",
            "te": "వేపనూనెతో ఆఫిడ్లను నియంత్రించి, సోకిన వేలను తొలగించండి.",
        },
    },
    "Blueberry___healthy": {
        "display": {"en": "Blueberry Healthy", "hi": "स्वस्थ ब्लूबेरी", "te": "ఆరోగ్యకరమైన బ్లూబెర్రీ"},
        "advice": {
            "en": "Apply acidic fertilizers such as ammonium sulfate.",
            "hi": "अमोनियम सल्फेट जैसे अम्लीय उर्वरक दें।",
            "te": "అమోనియం సల్ఫేట్ వంటి ఆమ్ల ధర్మం గల ఎరువులు వాడండి.",
        },
    },
    "Bottle gourd__Anthracnose": {
        "display": {"en": "Bottle Gourd Anthracnose", "hi": "लौकी का एन्थ्रेक्नोज", "te": "సొరకాయ ఆంత్రాక్నోస్"},
        "advice": {
            "en": "Apply Chlorothalonil or Mancozeb.",
            "hi": "क्लोरोथैलोनिल या मैनकोजेब का उपयोग करें।",
            "te": "క్లోరోథాలోనిల్ లేదా మ్యాంకోజెబ్ వాడండి.",
        },
    },
    "Bottle gourd__Downey_mildew": {
        "display": {"en": "Bottle Gourd Downy Mildew", "hi": "लौकी का डाउनी मिल्ड्यू", "te": "సొరకాయ డౌనీ మిల్డ్యూ"},
        "advice": {
            "en": "Use Metalaxyl or copper-based fungicides.",
            "hi": "मेटालेक्सिल या तांबा-आधारित फफूंदनाशक उपयोग करें।",
            "te": "మెటాలాక్సిల్ లేదా రాగి ఆధారిత ఫంగిసైడ్లు వాడండి.",
        },
    },
    "Bottle gourd__Fresh_leaf": {
        "display": {"en": "Bottle Gourd Fresh Leaf", "hi": "ताज़ी लौकी पत्ती", "te": "తాజా సొరకాయ ఆకు"},
        "advice": {
            "en": "Apply well-rotted farmyard manure.",
            "hi": "अच्छी तरह सड़ी हुई गोबर की खाद दें।",
            "te": "బాగా కుళ్లిన పశువుల ఎరువును వేయండి.",
        },
    },
    "Cauliflower__Black_Rot": {
        "display": {"en": "Cauliflower Black Rot", "hi": "फूलगोभी का ब्लैक रॉट", "te": "కాలిఫ్లవర్ బ్లాక్ రాట్"},
        "advice": {
            "en": "Treat seeds with Streptocycline and spray Copper Oxychloride.",
            "hi": "बीजों को स्ट्रेप्टोसाइक्लिन से उपचारित करें और कॉपर ऑक्सीक्लोराइड स्प्रे करें।",
            "te": "విత్తనాలను స్ట్రెప్టోసైక్లిన్‌తో శుద్ధి చేసి కాపర్ ఆక్సీక్లోరైడ్ స్ప్రే చేయండి.",
        },
    },
    "Cauliflower__Downy_mildew": {
        "display": {"en": "Cauliflower Downy Mildew", "hi": "फूलगोभी का डाउनी मिल्ड्यू", "te": "కాలిఫ్లవర్ డౌనీ మిల్డ్యూ"},
        "advice": {
            "en": "Apply Ridomil Gold or Mancozeb.",
            "hi": "रिडोमिल गोल्ड या मैनकोजेब का उपयोग करें।",
            "te": "రిడోమిల్ గోల్డ్ లేదా మ్యాంకోజెబ్ వాడండి.",
        },
    },
    "Cauliflower__Fresh_leaf": {
        "display": {"en": "Cauliflower Fresh Leaf", "hi": "ताज़ी फूलगोभी पत्ती", "te": "తాజా కాలిఫ్లవర్ ఆకు"},
        "advice": {
            "en": "Top-dress with urea for leafy growth.",
            "hi": "पत्तेदार वृद्धि के लिए यूरिया टॉप ड्रेसिंग करें।",
            "te": "ఆకుల పెరుగుదల కోసం యూరియా టాప్ డ్రెస్సింగ్ చేయండి.",
        },
    },
    "Cherry___Powdery_mildew": {
        "display": {"en": "Cherry Powdery Mildew", "hi": "चेरी का पाउडरी मिल्ड्यू", "te": "చెర్రీ పౌడరీ మిల్డ్యూ"},
        "advice": {
            "en": "Use wettable sulfur or myclobutanil sprays.",
            "hi": "वेटेबल सल्फर या मायक्लोब्यूटानिल स्प्रे करें।",
            "te": "వెట్టబుల్ సల్ఫర్ లేదా మైక్లోబ్యూటానిల్ స్ప్రే చేయండి.",
        },
    },
    "Cherry___healthy": {
        "display": {"en": "Cherry Healthy", "hi": "स्वस्थ चेरी", "te": "ఆరోగ్యకరమైన చెర్రీ"},
        "advice": {
            "en": "Apply potassium-rich fertilizers during fruiting.",
            "hi": "फलधारण के समय पोटैशियम युक्त उर्वरक दें।",
            "te": "పండ్ల దశలో పొటాషియం అధికంగా ఉన్న ఎరువులు వాడండి.",
        },
    },
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "display": {"en": "Corn Cercospora Gray Leaf Spot", "hi": "मक्का का ग्रे लीफ स्पॉट", "te": "మొక్కజొన్న గ్రే లీఫ్ స్పాట్"},
        "advice": {
            "en": "Apply propiconazole fungicide.",
            "hi": "प्रोपिकोनाजोल फफूंदनाशक का प्रयोग करें।",
            "te": "ప్రోపికోనాజోల్ ఫంగిసైడ్ వాడండి.",
        },
    },
    "Corn___Common_rust": {
        "display": {"en": "Corn Common Rust", "hi": "मक्का का कॉमन रस्ट", "te": "మొక్కజొన్న కామన్ రస్ట్"},
        "advice": {
            "en": "Spray Mancozeb (0.2%) on foliage.",
            "hi": "पत्तियों पर मैनकोजेब (0.2%) स्प्रे करें।",
            "te": "ఆకులపై మ్యాంకోజెబ్ (0.2%) స్ప్రే చేయండి.",
        },
    },
    "Corn___Northern_Leaf_Blight": {
        "display": {"en": "Corn Northern Leaf Blight", "hi": "मक्का का नॉर्दर्न लीफ ब्लाइट", "te": "మొక్కజొన్న నార్తర్న్ లీఫ్ బ్లైట్"},
        "advice": {
            "en": "Use resistant hybrids and apply Azoxystrobin.",
            "hi": "प्रतिरोधी किस्में लगाएँ और एजोक्सीस्ट्रोबिन का प्रयोग करें।",
            "te": "నిరోధక హైబ్రిడ్లు వాడి అజోక్సీస్ట్రోబిన్ ప్రయోగించండి.",
        },
    },
    "Corn___healthy": {
        "display": {"en": "Corn Healthy", "hi": "स्वस्थ मक्का", "te": "ఆరోగ్యకరమైన మొక్కజొన్న"},
        "advice": {
            "en": "Side-dress with nitrogen at knee height.",
            "hi": "घुटने की ऊँचाई पर नाइट्रोजन की साइड ड्रेसिंग करें।",
            "te": "మొక్క మోకాలివరకు పెరిగినప్పుడు నైట్రోజన్ సైడ్ డ్రెస్సింగ్ చేయండి.",
        },
    },
    "Cucumber__Anthracnose_lesions": {
        "display": {"en": "Cucumber Anthracnose", "hi": "खीरे का एन्थ्रेक्नोज", "te": "దోసకాయ ఆంత్రాక్నోస్"},
        "advice": {
            "en": "Apply Chlorothalonil and avoid overhead irrigation.",
            "hi": "क्लोरोथैलोनिल का उपयोग करें और ऊपर से सिंचाई न करें।",
            "te": "క్లోరోథాలోనిల్ వాడి పై నుంచి నీరు పోయడం నివారించండి.",
        },
    },
    "Cucumber__Downy_mildew": {
        "display": {"en": "Cucumber Downy Mildew", "hi": "खीरे का डाउनी मिल्ड्यू", "te": "దోసకాయ డౌనీ మిల్డ్యూ"},
        "advice": {
            "en": "Use systemic fungicides such as Metalaxyl.",
            "hi": "मेटालेक्सिल जैसे सिस्टमेटिक फफूंदनाशक का उपयोग करें।",
            "te": "మెటాలాక్సిల్ వంటి సిస్టమిక్ ఫంగిసైడ్లు వాడండి.",
        },
    },
    "Cucumber__Fresh_leaf": {
        "display": {"en": "Cucumber Fresh Leaf", "hi": "ताज़ी खीरा पत्ती", "te": "తాజా దోసకాయ ఆకు"},
        "advice": {
            "en": "Apply liquid seaweed fertilizer.",
            "hi": "लिक्विड सीवीड उर्वरक दें।",
            "te": "ద్రవరూప సముద్రశైవల ఎరువు వేయండి.",
        },
    },
    "Eggplant_Cercopora_leaf_spot": {
        "display": {"en": "Eggplant Cercospora Leaf Spot", "hi": "बैंगन का सर्कोस्पोरा लीफ स्पॉट", "te": "వంకాయ సెర్కోస్పోరా లీఫ్ స్పాట్"},
        "advice": {
            "en": "Spray Mancozeb or Zineb every 10 days.",
            "hi": "हर 10 दिन में मैनकोजेब या जिनेब स्प्रे करें।",
            "te": "ప్రతి 10 రోజులకు మ్యాంకోజెబ్ లేదా జినెబ్ స్ప్రే చేయండి.",
        },
    },
    "Eggplant_begomovirus": {
        "display": {"en": "Eggplant Begomovirus", "hi": "बैंगन बेगोमोवायरस", "te": "వంకాయ బేగోమోవైరస్"},
        "advice": {
            "en": "Control whiteflies using Imidacloprid.",
            "hi": "इमिडाक्लोप्रिड से सफेद मक्खी नियंत्रित करें।",
            "te": "ఇమిడాక్లోప్రిడ్‌తో వైట్‌ఫ్లై నియంత్రించండి.",
        },
    },
    "Eggplant_fresh_leaf": {
        "display": {"en": "Eggplant Fresh Leaf", "hi": "ताज़ी बैंगन पत्ती", "te": "తాజా వంకాయ ఆకు"},
        "advice": {
            "en": "Apply NPK 15:15:15.",
            "hi": "NPK 15:15:15 का प्रयोग करें।",
            "te": "NPK 15:15:15 వాడండి.",
        },
    },
    "Eggplant_verticillium_wilt": {
        "display": {"en": "Eggplant Verticillium Wilt", "hi": "बैंगन का वर्टिसिलियम विल्ट", "te": "వంకాయ వెర్టిసిల్లియం విల్ట్"},
        "advice": {
            "en": "Use soil solarization and Trichoderma viride.",
            "hi": "मिट्टी का सौर उपचार करें और ट्राइकोडर्मा विरिडे का उपयोग करें।",
            "te": "మట్టి సోలరైజేషన్ చేసి ట్రైకోడెర్మా విరిడే వాడండి.",
        },
    },
    "Grape___Black_rot": {
        "display": {"en": "Grape Black Rot", "hi": "अंगूर का ब्लैक रॉट", "te": "ద్రాక్ష బ్లాక్ రాట్"},
        "advice": {
            "en": "Spray Captan or Mancozeb at bloom stage.",
            "hi": "फूल आने की अवस्था में कैप्टान या मैनकोजेब स्प्रे करें।",
            "te": "పుష్పదశలో క్యాప్టాన్ లేదా మ్యాంకోజెబ్ స్ప్రే చేయండి.",
        },
    },
    "Grape___Esca_(Black_Measles)": {
        "display": {"en": "Grape Esca (Black Measles)", "hi": "अंगूर एस्का (ब्लैक मीज़ल्स)", "te": "ద్రాక్ష ఎస్కా (బ్లాక్ మీజిల్స్)"},
        "advice": {
            "en": "Protect pruning wounds with fungicides.",
            "hi": "छंटाई के घावों को फफूंदनाशक से सुरक्षित रखें।",
            "te": "కత్తిరింపు గాయాలను ఫంగిసైడ్లతో రక్షించండి.",
        },
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display": {"en": "Grape Leaf Blight", "hi": "अंगूर पत्ती झुलसा", "te": "ద్రాక్ష లీఫ్ బ్లైట్"},
        "advice": {
            "en": "Spray Bordeaux mixture (1%).",
            "hi": "बोर्डो मिश्रण (1%) का छिड़काव करें।",
            "te": "బోర్డో మిశ్రమం (1%) స్ప్రే చేయండి.",
        },
    },
    "Grape___healthy": {
        "display": {"en": "Grape Healthy", "hi": "स्वस्थ अंगूर", "te": "ఆరోగ్యకరమైన ద్రాక్ష"},
        "advice": {
            "en": "Apply muriate of potash for better sugar content.",
            "hi": "शर्करा बढ़ाने के लिए म्युरेट ऑफ पोटाश दें।",
            "te": "చక్కెర శాతం మెరుగుపడేందుకు మ్యూయరేట్ ఆఫ్ పొటాష్ వాడండి.",
        },
    },
    "Guava_Healthy": {
        "display": {"en": "Guava Healthy", "hi": "स्वस्थ अमरूद", "te": "ఆరోగ్యకరమైన జామ"},
        "advice": {
            "en": "Apply NPK 6:6:6 and zinc micronutrient spray.",
            "hi": "NPK 6:6:6 और जिंक सूक्ष्म पोषक स्प्रे दें।",
            "te": "NPK 6:6:6 తో పాటు జింక్ సూక్ష్మపోషక స్ప్రే చేయండి.",
        },
    },
    "Guava_Phytopthora": {
        "display": {"en": "Guava Phytophthora", "hi": "अमरूद फाइटोफ्थोरा", "te": "జామ ఫైటోఫ్తోరా"},
        "advice": {
            "en": "Improve drainage and apply Fosetyl-Al.",
            "hi": "जल निकास सुधारें और फोसेटिल-एएल का प्रयोग करें।",
            "te": "డ్రైనేజీ మెరుగుపరచి ఫోసెటిల్-అల్ వాడండి.",
        },
    },
    "Guava_Red_rust": {
        "display": {"en": "Guava Red Rust", "hi": "अमरूद रेड रस्ट", "te": "జామ రెడ్ రస్ట్"},
        "advice": {
            "en": "Spray Copper Oxychloride (0.3%).",
            "hi": "कॉपर ऑक्सीक्लोराइड (0.3%) स्प्रे करें।",
            "te": "కాపర్ ఆక్సీక్లోరైడ్ (0.3%) స్ప్రే చేయండి.",
        },
    },
    "Guava_Scab": {
        "display": {"en": "Guava Scab", "hi": "अमरूद स्कैब", "te": "జామ స్క్యాబ్"},
        "advice": {
            "en": "Spray Carbendazim on foliage.",
            "hi": "पत्तियों पर कार्बेन्डाजिम स्प्रे करें।",
            "te": "ఆకులపై కార్బెండాజిం స్ప్రే చేయండి.",
        },
    },
    "Guava_Styler_and_Root": {
        "display": {"en": "Guava Styler and Root Disorder", "hi": "अमरूद स्टाइलर और जड़ विकार", "te": "జామ స్టైలర్ మరియు వేరు సమస్య"},
        "advice": {
            "en": "Apply zinc and boron to the soil.",
            "hi": "मिट्टी में जिंक और बोरॉन दें।",
            "te": "మట్టిలో జింక్ మరియు బోరాన్ ఇవ్వండి.",
        },
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display": {"en": "Orange Citrus Greening", "hi": "संतरे का सिट्रस ग्रीनिंग", "te": "నారింజ సిట్రస్ గ్రీనింగ్"},
        "advice": {
            "en": "Control citrus psyllid and apply micronutrients.",
            "hi": "सिट्रस सायलिड नियंत्रित करें और सूक्ष्म पोषक दें।",
            "te": "సిట్రస్ సైలిడ్‌ను నియంత్రించి సూక్ష్మపోషకాలు ఇవ్వండి.",
        },
    },
    "Paddy_Bacterial_leaf_blight": {
        "display": {"en": "Paddy Bacterial Leaf Blight", "hi": "धान का बैक्टीरियल लीफ ब्लाइट", "te": "వరి బ్యాక్టీరియల్ లీఫ్ బ్లైట్"},
        "advice": {
            "en": "Spray Streptocycline with Copper Oxychloride.",
            "hi": "स्ट्रेप्टोसाइक्लिन और कॉपर ऑक्सीक्लोराइड स्प्रे करें।",
            "te": "స్ట్రెప్టోసైక్లిన్‌తో కాపర్ ఆక్సీక్లోరైడ్ స్ప్రే చేయండి.",
        },
    },
    "Paddy_Brown_spot": {
        "display": {"en": "Paddy Brown Spot", "hi": "धान का ब्राउन स्पॉट", "te": "వరి బ్రౌన్ స్పాట్"},
        "advice": {
            "en": "Apply potash and spray Edifenphos or Mancozeb.",
            "hi": "पोटाश दें और एडिफेनफॉस या मैनकोजेब स्प्रे करें।",
            "te": "పొటాష్ ఇచ్చి ఎడిఫెన్‌ఫాస్ లేదా మ్యాంకోజెబ్ స్ప్రే చేయండి.",
        },
    },
    "Paddy_Leaf_smut": {
        "display": {"en": "Paddy Leaf Smut", "hi": "धान लीफ स्मट", "te": "వరి లీఫ్ స్మట్"},
        "advice": {
            "en": "Spray Propiconazole (0.1%).",
            "hi": "प्रोपिकोनाजोल (0.1%) स्प्रे करें।",
            "te": "ప్రోపికోనాజోల్ (0.1%) స్ప్రే చేయండి.",
        },
    },
    "Peach___Bacterial_spot": {
        "display": {"en": "Peach Bacterial Spot", "hi": "आड़ू का बैक्टीरियल स्पॉट", "te": "పీచ్ బ్యాక్టీరియల్ స్పాట్"},
        "advice": {
            "en": "Use copper sprays during dormancy.",
            "hi": "सुप्तावस्था में कॉपर स्प्रे करें।",
            "te": "సుషుప్త దశలో కాపర్ స్ప్రే చేయండి.",
        },
    },
    "Peach___healthy": {
        "display": {"en": "Peach Healthy", "hi": "स्वस्थ आड़ू", "te": "ఆరోగ్యకరమైన పీచ్"},
        "advice": {
            "en": "Apply balanced NPK in early spring.",
            "hi": "प्रारंभिक वसंत में संतुलित NPK दें।",
            "te": "వసంత ఆరంభంలో సంతులిత NPK ఇవ్వండి.",
        },
    },
    "Pepper_bell___Bacterial_spot": {
        "display": {"en": "Bell Pepper Bacterial Spot", "hi": "शिमला मिर्च का बैक्टीरियल स्पॉट", "te": "బెల్ పెప్పర్ బ్యాక్టీరియల్ స్పాట్"},
        "advice": {
            "en": "Use copper hydroxide sprays and disease-free seeds.",
            "hi": "कॉपर हाइड्रॉक्साइड स्प्रे करें और रोगमुक्त बीज लें।",
            "te": "కాపర్ హైడ్రాక్సైడ్ స్ప్రే చేసి, రోగరహిత విత్తనాలు వాడండి.",
        },
    },
    "Pepper_bell___healthy": {
        "display": {"en": "Bell Pepper Healthy", "hi": "स्वस्थ शिमला मिर्च", "te": "ఆరోగ్యకరమైన బెల్ పెప్పర్"},
        "advice": {
            "en": "Apply 5-10-10 NPK fertilizer.",
            "hi": "5-10-10 NPK उर्वरक दें।",
            "te": "5-10-10 NPK ఎరువు వాడండి.",
        },
    },
    "Potato___Early_blight": {
        "display": {"en": "Potato Early Blight", "hi": "आलू का अर्ली ब्लाइट", "te": "బంగాళాదుంప ఎర్లీ బ్లైట్"},
        "advice": {
            "en": "Spray Mancozeb or Chlorothalonil.",
            "hi": "मैनकोजेब या क्लोरोथैलोनिल स्प्रे करें।",
            "te": "మ్యాంకోజెబ్ లేదా క్లోరోథాలోనిల్ స్ప్రే చేయండి.",
        },
    },
    "Potato___Late_blight": {
        "display": {"en": "Potato Late Blight", "hi": "आलू का लेट ब्लाइट", "te": "బంగాళాదుంప లేట్ బ్లైట్"},
        "advice": {
            "en": "Spray Ridomil or Metalaxyl-M with Mancozeb.",
            "hi": "रिडोमिल या मेटालेक्सिल-एम के साथ मैनकोजेब स्प्रे करें।",
            "te": "రిడోమిల్ లేదా మెటాలాక్సిల్-ఎం తో మ్యాంకోజెబ్ స్ప్రే చేయండి.",
        },
    },
    "Potato___healthy": {
        "display": {"en": "Potato Healthy", "hi": "स्वस्थ आलू", "te": "ఆరోగ్యకరమైన బంగాళాదుంప"},
        "advice": {
            "en": "Apply high nitrogen during early growth.",
            "hi": "प्रारंभिक वृद्धि में नाइट्रोजन अधिक दें।",
            "te": "ప్రారంభ పెరుగుదలలో అధిక నైట్రోజన్ ఇవ్వండి.",
        },
    },
    "Raspberry___healthy": {
        "display": {"en": "Raspberry Healthy", "hi": "स्वस्थ रास्पबेरी", "te": "ఆరోగ్యకరమైన రాస్ప్‌బెర్రీ"},
        "advice": {
            "en": "Apply 10-10-10 NPK in spring.",
            "hi": "वसंत ऋतु में 10-10-10 NPK दें।",
            "te": "వసంత కాలంలో 10-10-10 NPK ఇవ్వండి.",
        },
    },
    "Soybean___healthy": {
        "display": {"en": "Soybean Healthy", "hi": "स्वस्थ सोयाबीन", "te": "ఆరోగ్యకరమైన సోయాబీన్"},
        "advice": {
            "en": "Apply phosphorus and Rhizobium inoculation.",
            "hi": "फॉस्फोरस दें और राइजोबियम इनोकुलेशन करें।",
            "te": "ఫాస్ఫరస్ ఇవ్వండి మరియు రైజోబియం ఇనాక్యులేషన్ చేయండి.",
        },
    },
    "Squash___Powdery_mildew": {
        "display": {"en": "Squash Powdery Mildew", "hi": "स्क्वैश का पाउडरी मिल्ड्यू", "te": "స్క్వాష్ పౌడరీ మిల్డ్యూ"},
        "advice": {
            "en": "Use neem oil or sulfur sprays.",
            "hi": "नीम तेल या सल्फर स्प्रे का उपयोग करें।",
            "te": "వేపనూనె లేదా సల్ఫర్ స్ప్రే వాడండి.",
        },
    },
    "Strawberry___Leaf_scorch": {
        "display": {"en": "Strawberry Leaf Scorch", "hi": "स्ट्रॉबेरी लीफ स्कॉर्च", "te": "స్ట్రాబెర్రీ లీఫ్ స్కార్చ్"},
        "advice": {
            "en": "Avoid excess nitrogen and apply copper fungicides.",
            "hi": "अधिक नाइट्रोजन से बचें और कॉपर फफूंदनाशक दें।",
            "te": "అధిక నైట్రోజన్ నివారించి కాపర్ ఫంగిసైడ్లు వాడండి.",
        },
    },
    "Strawberry___healthy": {
        "display": {"en": "Strawberry Healthy", "hi": "स्वस्थ स्ट्रॉबेरी", "te": "ఆరోగ్యకరమైన స్ట్రాబెర్రీ"},
        "advice": {
            "en": "Use phosphorus-rich fertilizer for berries.",
            "hi": "बेरी उत्पादन के लिए फॉस्फोरस युक्त उर्वरक दें।",
            "te": "పండ్ల కోసం ఫాస్ఫరస్ అధికంగా ఉన్న ఎరువు వాడండి.",
        },
    },
    "Sugarcane_Healthy": {
        "display": {"en": "Sugarcane Healthy", "hi": "स्वस्थ गन्ना", "te": "ఆరోగ్యకరమైన చెరకు"},
        "advice": {
            "en": "Apply balanced NPK and add iron or zinc if yellowing appears.",
            "hi": "संतुलित NPK दें और पीलापन हो तो आयरन/जिंक दें।",
            "te": "సంతులిత NPK ఇవ్వండి; పసుపు కనిపిస్తే ఇనుము లేదా జింక్ ఇవ్వండి.",
        },
    },
    "Sugarcane_Mosaic": {
        "display": {"en": "Sugarcane Mosaic", "hi": "गन्ना मोज़ेक", "te": "చెరకు మోసాయిక్"},
        "advice": {
            "en": "Use virus-free setts and control aphids.",
            "hi": "वायरस-मुक्त सेट्स उपयोग करें और एफिड नियंत्रित करें।",
            "te": "వైరస్-రహిత సెట్లను వాడి ఆఫిడ్లను నియంత్రించండి.",
        },
    },
    "Sugarcane_RedRot": {
        "display": {"en": "Sugarcane Red Rot", "hi": "गन्ना रेड रॉट", "te": "చెరకు రెడ్ రాట్"},
        "advice": {
            "en": "Treat setts with Carbendazim and improve drainage.",
            "hi": "सेट्स को कार्बेन्डाजिम से उपचारित करें और जल निकास सुधारें।",
            "te": "సెట్లను కార్బెండాజిం‌తో శుద్ధి చేసి డ్రైనేజీ మెరుగుపరచండి.",
        },
    },
    "Sugarcane_Rust": {
        "display": {"en": "Sugarcane Rust", "hi": "गन्ना रस्ट", "te": "చెరకు రస్ట్"},
        "advice": {
            "en": "Spray Pyraclostrobin or Mancozeb.",
            "hi": "पाइराक्लोस्ट्रोबिन या मैनकोजेब स्प्रे करें।",
            "te": "పైరాక్లోస్ట్రోబిన్ లేదా మ్యాంకోజెబ్ స్ప్రే చేయండి.",
        },
    },
    "Sugarcane_Yellow": {
        "display": {"en": "Sugarcane Yellow Disease", "hi": "गन्ना येलो रोग", "te": "చెరకు పసుపు వ్యాధి"},
        "advice": {
            "en": "Apply ferrous sulfate (0.5%) as foliar spray.",
            "hi": "फेरस सल्फेट (0.5%) का पर्णीय स्प्रे करें।",
            "te": "ఫెరస్ సల్ఫేట్ (0.5%) ఆకులపై స్ప్రే చేయండి.",
        },
    },
    "Tomato___Bacterial_spot": {
        "display": {"en": "Tomato Bacterial Spot", "hi": "टमाटर का बैक्टीरियल स्पॉट", "te": "టమాటా బ్యాక్టీరియల్ స్పాట్"},
        "advice": {
            "en": "Use copper spray with Streptocycline.",
            "hi": "कॉपर स्प्रे के साथ स्ट्रेप्टोसाइक्लिन का उपयोग करें।",
            "te": "కాపర్ స్ప్రేతో స్ట్రెప్టోసైక్లిన్ ఉపయోగించండి.",
        },
    },
    "Tomato___Early_blight": {
        "display": {"en": "Tomato Early Blight", "hi": "टमाटर अर्ली ब्लाइट", "te": "టమాటా ఎర్లీ బ్లైట్"},
        "advice": {
            "en": "Apply Chlorothalonil or Azoxystrobin.",
            "hi": "क्लोरोथैलोनिल या एजोक्सीस्ट्रोबिन का प्रयोग करें।",
            "te": "క్లోరోథాలోనిల్ లేదా అజోక్సీస్ట్రోబిన్ వాడండి.",
        },
    },
    "Tomato___Late_blight": {
        "display": {"en": "Tomato Late Blight", "hi": "टमाटर लेट ब्लाइट", "te": "టమాటా లేట్ బ్లైట్"},
        "advice": {
            "en": "Spray Mancozeb or Ridomil.",
            "hi": "मैनकोजेब या रिडोमिल स्प्रे करें।",
            "te": "మ్యాంకోజెబ్ లేదా రిడోమిల్ స్ప్రే చేయండి.",
        },
    },
    "Tomato___Leaf_Mold": {
        "display": {"en": "Tomato Leaf Mold", "hi": "टमाटर लीफ मोल्ड", "te": "టమాటా లీఫ్ మోల్డ్"},
        "advice": {
            "en": "Increase ventilation and apply copper fungicide.",
            "hi": "वेंटिलेशन बढ़ाएँ और कॉपर फफूंदनाशक लगाएँ।",
            "te": "గాలి ప్రవాహం పెంచి కాపర్ ఫంగిసైడ్ వాడండి.",
        },
    },
    "Tomato___Septoria_leaf_spot": {
        "display": {"en": "Tomato Septoria Leaf Spot", "hi": "टमाटर सेप्टोरिया लीफ स्पॉट", "te": "టమాటా సెప్టోరియా లీఫ్ స్పాట్"},
        "advice": {
            "en": "Use Chlorothalonil or Mancozeb sprays.",
            "hi": "क्लोरोथैलोनिल या मैनकोजेब स्प्रे करें।",
            "te": "క్లోరోథాలోనిల్ లేదా మ్యాంకోజెబ్ స్ప్రే చేయండి.",
        },
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display": {"en": "Tomato Spider Mites", "hi": "टमाटर स्पाइडर माइट", "te": "టమాటా స్పైడర్ మైట్స్"},
        "advice": {
            "en": "Apply Abamectin or neem oil spray.",
            "hi": "अबामेक्टिन या नीम तेल स्प्रे करें।",
            "te": "అబామెక్టిన్ లేదా వేపనూనె స్ప్రే చేయండి.",
        },
    },
    "Tomato___Target_Spot": {
        "display": {"en": "Tomato Target Spot", "hi": "टमाटर टारगेट स्पॉट", "te": "టమాటా టార్గెట్ స్పాట్"},
        "advice": {
            "en": "Apply Pyraclostrobin or Boscalid.",
            "hi": "पाइराक्लोस्ट्रोबिन या बोस्कालिड का प्रयोग करें।",
            "te": "పైరాక్లోస్ట్రోబిన్ లేదా బోస్కాలిడ్ వాడండి.",
        },
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display": {"en": "Tomato Yellow Leaf Curl Virus", "hi": "टमाटर येलो लीफ कर्ल वायरस", "te": "టమాటా యెల్లో లీఫ్ కర్ల్ వైరస్"},
        "advice": {
            "en": "Use yellow sticky traps and control whiteflies.",
            "hi": "पीले स्टिकी ट्रैप लगाएँ और सफेद मक्खी नियंत्रित करें।",
            "te": "పసుపు స్టిక్కీ ట్రాప్లు పెట్టి వైట్‌ఫ్లై నియంత్రించండి.",
        },
    },
    "Tomato___Tomato_mosaic_virus": {
        "display": {"en": "Tomato Mosaic Virus", "hi": "टमाटर मोज़ेक वायरस", "te": "టమాటా మోసాయిక్ వైరస్"},
        "advice": {
            "en": "Sanitize tools and use resistant varieties.",
            "hi": "उपकरण साफ रखें और प्रतिरोधी किस्में लगाएँ।",
            "te": "పరికరాలు శుభ్రంగా ఉంచి నిరోధక రకాలు వాడండి.",
        },
    },
    "Tomato___healthy": {
        "display": {"en": "Tomato Healthy", "hi": "स्वस्थ टमाटर", "te": "ఆరోగ్యకరమైన టమాటా"},
        "advice": {
            "en": "Use regular NPK 10-10-10 and calcium.",
            "hi": "नियमित NPK 10-10-10 और कैल्शियम दें।",
            "te": "సాధారణ NPK 10-10-10 తో పాటు కాల్షియం ఇవ్వండి.",
        },
    },
    "Wheat_Healthy": {
        "display": {"en": "Wheat Healthy", "hi": "स्वस्थ गेहूँ", "te": "ఆరోగ్యకరమైన గోధుమ"},
        "advice": {
            "en": "Apply DAP and top-dress with urea.",
            "hi": "DAP दें और ऊपर से यूरिया डालें।",
            "te": "DAP ఇచ్చి యూరియా టాప్ డ్రెస్సింగ్ చేయండి.",
        },
    },
    "Wheat_leaf_leaf_stripe_rust": {
        "display": {"en": "Wheat Stripe Rust", "hi": "गेहूँ स्ट्राइप रस्ट", "te": "గోధుమ స్ట్రైప్ రస్ట్"},
        "advice": {
            "en": "Apply Propiconazole 25 EC.",
            "hi": "प्रोपिकोनाजोल 25 EC का प्रयोग करें।",
            "te": "ప్రోపికోనాజోల్ 25 EC వాడండి.",
        },
    },
    "Wheatleaf_septoria": {
        "display": {"en": "Wheat Septoria", "hi": "गेहूँ सेप्टोरिया", "te": "గోధుమ సెప్టోరియా"},
        "advice": {
            "en": "Spray Tebuconazole or Chlorothalonil.",
            "hi": "टेबुकोनाजोल या क्लोरोथैलोनिल स्प्रे करें।",
            "te": "టెబ్యుకోనాజోల్ లేదా క్లోరోథాలోనిల్ స్ప్రే చేయండి.",
        },
    },
}

DEFAULT_DISEASE_FALLBACK = {
    key: {
        "display": {"en": key.replace("___", " ").replace("__", " ").replace("_", " ")},
        "advice": {"en": "Follow integrated crop management practices."},
    }
    for key in DISEASE_METADATA
}

DEFAULT_GRADCAM_LAYER = "mobilenetv2_1.00_224/out_relu"
GRADCAM_CACHE_VERSION = "gradcam_v3"

DISEASE_METADATA_ALIASES = {
    "Bitter Gourd___Downey_mildew": "Bitter Gourd__Downy_mildew",
    "Bitter Gourd___Fresh_leaf": "Bitter Gourd__Fresh_leaf",
    "Bitter Gourd___Fusarium_wilt": "Bitter Gourd__Fusarium_wilt",
    "Bitter Gourd___Mosaic_virus": "Bitter Gourd__Mosaic_virus",
    "Bottle gourd___Anthracnose": "Bottle gourd__Anthracnose",
    "Bottle gourd___Downey_mildew": "Bottle gourd__Downey_mildew",
    "Bottle gourd___Fresh_leaf": "Bottle gourd__Fresh_leaf",
    "Cauliflower___Black_Rot": "Cauliflower__Black_Rot",
    "Cauliflower___Downy_mildew": "Cauliflower__Downy_mildew",
    "Cauliflower___Fresh_leaf": "Cauliflower__Fresh_leaf",
    "Cucumber___Anthracnose_lesions": "Cucumber__Anthracnose_lesions",
    "Cucumber___Downy_mildew": "Cucumber__Downy_mildew",
    "Cucumber___Fresh_leaf": "Cucumber__Fresh_leaf",
    "Pepper,_bell___Bacterial_spot": "Pepper_bell___Bacterial_spot",
    "Pepper,_bell___healthy": "Pepper_bell___healthy",
    "Sugarcae_Healthy": "Sugarcane_Healthy",
    "Sugarcae_Mosaic": "Sugarcane_Mosaic",
    "Sugarcae_RedRot": "Sugarcane_RedRot",
    "Sugarcae_Rust": "Sugarcane_Rust",
    "Sugarcae_Yellow": "Sugarcane_Yellow",
    "Wheat_leaf_septoria": "Wheatleaf_septoria",
    "Wheat_leaf_stripe_rust": "Wheat_leaf_leaf_stripe_rust",
}


def load_class_names():
    class_names_path = Path(__file__).resolve().parent / "training_outputs" / "class_names.json"
    if class_names_path.exists():
        try:
            with open(class_names_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            pass
    return list(DISEASE_METADATA.keys())


CLASS_NAMES = load_class_names()

CROP_METADATA = {
    "rice": {
        "name": {"en": "Rice", "hi": "चावल", "te": "వరి"},
        "fertilizer": {
            "en": "Apply urea, DAP, and MOP in split doses and ensure timely irrigation.",
            "hi": "यूरिया, DAP और MOP को भागों में दें और समय पर सिंचाई करें।",
            "te": "యూరియా, DAP, MOP ను విడతలవారీగా ఇచ్చి సమయానికి నీరు పెట్టండి.",
        },
        "description": {
            "en": "Rice is a staple food crop grown in warm, humid climates, often in flooded fields.",
            "hi": "धान एक प्रमुख खाद्य फसल है जो गर्म और आर्द्र जलवायु में, अक्सर पानी भरे खेतों में उगाई जाती है।",
            "te": "వరి ఒక ప్రధాన ఆహార పంట. ఇది వేడి, తేమ గల వాతావరణంలో సాధారణంగా నీరు నిలిచే పొలాల్లో సాగు చేస్తారు.",
        },
        "conditions": {
            "en": "Best in clayey loam soil, 20-35°C temperatures, and high humidity.",
            "hi": "यह चिकनी दोमट मिट्टी, 20-35°C तापमान और अधिक आर्द्रता में अच्छी होती है।",
            "te": "ఇది మట్టి మిశ్రమం గల నేలలో, 20-35°C ఉష్ణోగ్రతలో మరియు అధిక ఆర్ద్రతలో బాగా పెరుగుతుంది.",
        },
        "tips": {
            "en": "Maintain standing water and use pest-resistant high-yield varieties.",
            "hi": "खेत में पानी बनाए रखें और कीट-रोधी उच्च उपज किस्में लगाएँ।",
            "te": "పొలంలో నీరు నిల్వ ఉంచి, పురుగు నిరోధక అధిక దిగుబడి రకాలు వాడండి.",
        },
    },
    "wheat": {
        "name": {"en": "Wheat", "hi": "गेहूँ", "te": "గోధుమ"},
        "fertilizer": {
            "en": "Use urea and DAP before sowing, then top-dress nitrogen at tillering.",
            "hi": "बुवाई से पहले यूरिया और DAP दें, फिर टिलरिंग पर नाइट्रोजन दें।",
            "te": "విత్తే ముందు యూరియా, DAP ఇవ్వండి; తరువాత టిల్లరింగ్ దశలో నైట్రోజన్ ఇవ్వండి.",
        },
        "description": {
            "en": "Wheat is a major cereal crop grown widely in temperate regions.",
            "hi": "गेहूँ एक प्रमुख अनाज फसल है जो समशीतोष्ण क्षेत्रों में व्यापक रूप से उगाई जाती है।",
            "te": "గోధుమ ఒక ప్రధాన ధాన్య పంట. ఇది సమశీతోష్ణ ప్రాంతాల్లో విస్తృతంగా పండుతుంది.",
        },
        "conditions": {
            "en": "Needs cool weather during growth and dry conditions during harvest.",
            "hi": "विकास के समय ठंडा मौसम और कटाई के समय शुष्क मौसम चाहिए।",
            "te": "పెరుగుదల దశలో చల్లని వాతావరణం, కోత దశలో ఎండగా ఉండే పరిస్థితులు అవసరం.",
        },
        "tips": {
            "en": "Use certified seeds and apply nitrogen in split doses.",
            "hi": "प्रमाणित बीज लें और नाइट्रोजन को भागों में दें।",
            "te": "ధృవీకరించిన విత్తనాలు వాడి, నైట్రోజన్‌ను విడతలవారీగా ఇవ్వండి.",
        },
    },
    "maize": {
        "name": {"en": "Maize", "hi": "मक्का", "te": "మొక్కజొన్న"},
        "fertilizer": {
            "en": "Use balanced NPK and apply nitrogen in two split doses.",
            "hi": "संतुलित NPK दें और नाइट्रोजन दो भागों में दें।",
            "te": "సంతులిత NPK ఇవ్వండి, నైట్రోజన్‌ను రెండు విడతలుగా వేయండి.",
        },
        "description": {
            "en": "Maize is used as food, fodder, and industrial raw material.",
            "hi": "मक्का भोजन, चारे और औद्योगिक कच्चे माल के रूप में उपयोग होता है।",
            "te": "మొక్కజొన్న ఆహారం, మేత మరియు పరిశ్రమల ముడి పదార్థంగా ఉపయోగపడుతుంది.",
        },
        "conditions": {
            "en": "Grows well at 21-27°C with moderate rainfall.",
            "hi": "यह 21-27°C तापमान और मध्यम वर्षा में अच्छी होती है।",
            "te": "21-27°C ఉష్ణోగ్రత మరియు మితమైన వర్షపాతంలో బాగా పెరుగుతుంది.",
        },
        "tips": {
            "en": "Ensure proper spacing and control weeds early.",
            "hi": "सही दूरी रखें और खरपतवार का जल्दी नियंत्रण करें।",
            "te": "సరైన దూరం పాటించి, మొలక దశలోనే కలుపు నియంత్రించండి.",
        },
    },
    "chickpea": {
        "name": {"en": "Chickpea", "hi": "चना", "te": "సెనగ"},
        "fertilizer": {
            "en": "Use Rhizobium inoculant and phosphorus-rich fertilizers.",
            "hi": "राइजोबियम इनोकुलेंट और फॉस्फोरस युक्त उर्वरक उपयोग करें।",
            "te": "రైజోబియం ఇనాక్యులెంట్ మరియు ఫాస్ఫరస్ అధికంగా ఉన్న ఎరువులు వాడండి.",
        },
        "description": {
            "en": "Chickpea is a protein-rich legume widely used in food.",
            "hi": "चना प्रोटीन से भरपूर दलहनी फसल है।",
            "te": "సెనగ ప్రోటీన్ సమృద్ధిగా కలిగిన పప్పు పంట.",
        },
        "conditions": {
            "en": "Thrives in well-drained sandy loam soils in semi-arid climates.",
            "hi": "अच्छी जल निकास वाली बलुई दोमट मिट्टी और अर्ध-शुष्क जलवायु में अच्छी होती है।",
            "te": "బాగా నీరు దిగే ఇసుక మిశ్రమ నేలలు, అర్ధశుష్క వాతావరణంలో బాగా పెరుగుతుంది.",
        },
        "tips": {
            "en": "Rotate with cereals to improve soil fertility.",
            "hi": "मिट्टी की उर्वरता बढ़ाने के लिए इसे अनाज फसलों के साथ चक्र में लें।",
            "te": "మట్టి సారాన్ని మెరుగుపరచేందుకు ధాన్య పంటలతో పంట మార్పిడి చేయండి.",
        },
    },
    "kidneybeans": {
        "name": {"en": "Kidney Beans", "hi": "राजमा", "te": "రాజ్మా"},
        "fertilizer": {
            "en": "Use balanced NPK with more phosphorus and well-rotted manure.",
            "hi": "संतुलित NPK में फॉस्फोरस अधिक दें और सड़ी हुई खाद का उपयोग करें।",
            "te": "ఫాస్ఫరస్ ఎక్కువగా ఉన్న సంతులిత NPK మరియు బాగా కుళ్లిన ఎరువు వాడండి.",
        },
        "description": {
            "en": "Kidney beans are rich in protein and fiber and prefer warm conditions.",
            "hi": "राजमा प्रोटीन और फाइबर से भरपूर है और गर्म परिस्थितियों में अच्छा होता है।",
            "te": "రాజ్మా ప్రోటీన్, ఫైబర్ అధికంగా కలిగి ఉండి వేడి పరిస్థితుల్లో బాగా పెరుగుతుంది.",
        },
        "conditions": {
            "en": "Performs best in warm climates with well-drained soils.",
            "hi": "गर्म जलवायु और अच्छी जल निकासी वाली मिट्टी में बेहतर होता है।",
            "te": "వేడి వాతావరణం, నీరు బాగా దిగే నేలల్లో ఉత్తమంగా పెరుగుతుంది.",
        },
        "tips": {
            "en": "Avoid waterlogging and provide support where needed.",
            "hi": "जलभराव से बचें और आवश्यकता होने पर सहारा दें।",
            "te": "నీరు నిల్వ కాకుండా చూసి, అవసరమైతే ఆధారం ఇవ్వండి.",
        },
    },
    "pigeonpeas": {
        "name": {"en": "Pigeon Peas", "hi": "अरहर", "te": "కందులు"},
        "fertilizer": {
            "en": "Apply DAP or SSP at sowing and add organic compost.",
            "hi": "बुवाई पर DAP या SSP दें और जैविक खाद मिलाएँ।",
            "te": "విత్తే సమయంలో DAP లేదా SSP ఇచ్చి సేంద్రీయ కంపోస్ట్ కలపండి.",
        },
        "description": {
            "en": "Pigeon pea is a drought-tolerant pulse crop.",
            "hi": "अरहर सूखा सहन करने वाली दलहनी फसल है।",
            "te": "కంది ఎండను తట్టుకునే పప్పు పంట.",
        },
        "conditions": {
            "en": "Grows in poor soils and prefers warm climates with moderate rainfall.",
            "hi": "यह कम उपजाऊ मिट्टी में भी उगती है और गर्म जलवायु पसंद करती है।",
            "te": "తక్కువ సారవంతమైన నేలల్లో కూడా పెరిగి, మితమైన వర్షపాతం ఉన్న వేడి వాతావరణం ఇష్టపడుతుంది.",
        },
        "tips": {
            "en": "Plant at the onset of rains and intercrop with cereals.",
            "hi": "बरसात की शुरुआत में बोएँ और अनाज फसलों के साथ मिलाएँ।",
            "te": "వర్షాకాలం ప్రారంభంలో విత్తి, ధాన్య పంటలతో కలిపి సాగు చేయండి.",
        },
    },
    "mothbeans": {
        "name": {"en": "Moth Beans", "hi": "मोंठ", "te": "మోత్ బీన్స్"},
        "fertilizer": {
            "en": "Use moderate NPK with emphasis on phosphorus and farmyard manure.",
            "hi": "मध्यम NPK दें, फॉस्फोरस पर जोर रखें और गोबर की खाद मिलाएँ।",
            "te": "మధ్యస్థ NPK తో పాటు ఫాస్ఫరస్‌పై దృష్టి పెట్టి పశువుల ఎరువు వాడండి.",
        },
        "description": {
            "en": "Moth beans are drought-tolerant legumes for arid regions.",
            "hi": "मोंठ शुष्क क्षेत्रों के लिए सूखा-सहनशील दलहनी फसल है।",
            "te": "మోత్ బీన్స్ ఎండప్రాంతాలకు అనుకూలమైన ఎండనిరోధక పప్పు పంట.",
        },
        "conditions": {
            "en": "Survives in sandy soils with minimal water.",
            "hi": "रेतीली मिट्टी और कम पानी में भी टिकती है।",
            "te": "ఇసుక నేలల్లో తక్కువ నీటితో కూడా జీవిస్తుంది.",
        },
        "tips": {
            "en": "Use early-maturing varieties in low-rainfall areas.",
            "hi": "कम वर्षा वाले क्षेत्रों में जल्दी पकने वाली किस्में लें।",
            "te": "తక్కువ వర్షపాతం ఉన్న ప్రాంతాల్లో త్వరగా పండే రకాలు వాడండి.",
        },
    },
    "mungbean": {
        "name": {"en": "Mung Bean", "hi": "मूंग", "te": "పెసలు"},
        "fertilizer": {
            "en": "Apply Rhizobium culture and balanced NPK; avoid excess nitrogen.",
            "hi": "राइजोबियम कल्चर और संतुलित NPK दें; अधिक नाइट्रोजन न दें।",
            "te": "రైజోబియం కల్చర్ మరియు సంతులిత NPK ఇవ్వండి; అధిక నైట్రోజన్ నివారించండి.",
        },
        "description": {
            "en": "Mung bean is a protein-rich pulse used in food and sprouts.",
            "hi": "मूंग प्रोटीन युक्त दलहनी फसल है, जिसे भोजन और अंकुरित रूप में उपयोग करते हैं।",
            "te": "పెసలు ప్రోటీన్ అధికంగా కలిగిన పప్పు పంట; ఆహారం, మొలకల రూపంలో వాడుతారు.",
        },
        "conditions": {
            "en": "Grows well in warm climates with moderate rainfall.",
            "hi": "गर्म जलवायु और मध्यम वर्षा में अच्छी होती है।",
            "te": "వేడి వాతావరణం, మితమైన వర్షపాతం ఉన్నప్పుడు బాగా పెరుగుతుంది.",
        },
        "tips": {
            "en": "Avoid waterlogging and rotate with cereals.",
            "hi": "जलभराव से बचें और अनाज फसलों के साथ चक्र अपनाएँ।",
            "te": "నీరు నిల్వ కాకుండా చూసి ధాన్య పంటలతో మార్పిడి చేయండి.",
        },
    },
    "blackgram": {
        "name": {"en": "Black Gram", "hi": "उड़द", "te": "మినుములు"},
        "fertilizer": {
            "en": "Use Rhizobium and phosphate-solubilizing bacteria with SSP.",
            "hi": "राइजोबियम, फॉस्फेट घुलनशील जीवाणु और SSP का उपयोग करें।",
            "te": "రైజోబియం, ఫాస్ఫేట్ ద్రావణ జీవాణువులు మరియు SSP వాడండి.",
        },
        "description": {
            "en": "Black gram is a pulse crop used in many traditional foods.",
            "hi": "उड़द एक महत्वपूर्ण दलहनी फसल है जिसका उपयोग कई पारंपरिक व्यंजनों में होता है।",
            "te": "మినుములు అనేక సంప్రదాయ ఆహారాలలో ఉపయోగించే ముఖ్యమైన పప్పు పంట.",
        },
        "conditions": {
            "en": "Thrives in warm, humid climates and loamy soils.",
            "hi": "गर्म, आर्द्र जलवायु और दोमट मिट्टी में अच्छी होती है।",
            "te": "వేడి, తేమ గల వాతావరణం మరియు లోమీ నేలల్లో బాగా పెరుగుతుంది.",
        },
        "tips": {
            "en": "Plant in well-drained soil and avoid excess irrigation.",
            "hi": "अच्छी जल निकासी वाली मिट्टी में बोएँ और अधिक सिंचाई न करें।",
            "te": "నీరు బాగా దిగే నేలలో విత్తి అధిక నీటిపారుదల నివారించండి.",
        },
    },
    "lentil": {
        "name": {"en": "Lentil", "hi": "मसूर", "te": "మసూర్"},
        "fertilizer": {
            "en": "Apply phosphorus and potassium and treat seeds with Rhizobium.",
            "hi": "फॉस्फोरस, पोटाश दें और बीज को राइजोबियम से उपचारित करें।",
            "te": "ఫాస్ఫరస్, పొటాష్ ఇవ్వండి మరియు విత్తనాలను రైజోబియంతో శుద్ధి చేయండి.",
        },
        "description": {
            "en": "Lentil is a valuable protein source in many diets.",
            "hi": "मसूर कई आहारों में प्रोटीन का महत्वपूर्ण स्रोत है।",
            "te": "మసూర్ అనేక ఆహారాలలో ప్రోటీన్‌కు ముఖ్య వనరు.",
        },
        "conditions": {
            "en": "Best suited to temperate climates with cool weather.",
            "hi": "समशीतोष्ण और ठंडे मौसम में बेहतर होती है।",
            "te": "సమశీతోష్ణ, చల్లని వాతావరణానికి అనుకూలంగా ఉంటుంది.",
        },
        "tips": {
            "en": "Choose disease-resistant varieties to reduce loss.",
            "hi": "हानि कम करने के लिए रोग-रोधी किस्में चुनें।",
            "te": "నష్టం తగ్గించేందుకు వ్యాధి నిరోధక రకాలు ఎంచుకోండి.",
        },
    },
    "pomegranate": {
        "name": {"en": "Pomegranate", "hi": "अनार", "te": "దానిమ్మ"},
        "fertilizer": {
            "en": "Use high-potassium fertilizer during fruit development and compost yearly.",
            "hi": "फल विकास के समय अधिक पोटाश दें और हर साल कम्पोस्ट डालें।",
            "te": "పండ్ల అభివృద్ధి దశలో అధిక పొటాషియం ఇవ్వండి, ప్రతి సంవత్సరం కంపోస్ట్ వేయండి.",
        },
        "description": {
            "en": "Pomegranate is a fruit crop valued for its sweet-tangy seeds.",
            "hi": "अनार मीठे-खट्टे दानों वाला मूल्यवान फल है।",
            "te": "దానిమ్మ తీపి-పులుపు గింజల కోసం విలువైన పండు.",
        },
        "conditions": {
            "en": "Prefers dry climates and sandy loam soils.",
            "hi": "शुष्क जलवायु और बलुई दोमट मिट्टी पसंद है।",
            "te": "ఎండగా ఉండే వాతావరణం, ఇసుక మిశ్రమ నేలలు ఇష్టపడుతుంది.",
        },
        "tips": {
            "en": "Prune regularly to maintain fruit quality.",
            "hi": "फल की गुणवत्ता बनाए रखने के लिए नियमित छँटाई करें।",
            "te": "పండ్ల నాణ్యత కోసం క్రమం తప్పకుండా కత్తిరింపు చేయండి.",
        },
    },
    "banana": {
        "name": {"en": "Banana", "hi": "केला", "te": "అరటి"},
        "fertilizer": {
            "en": "Banana needs high potassium, regular compost, and split nitrogen doses.",
            "hi": "केले को अधिक पोटाश, नियमित कम्पोस्ट और विभाजित नाइट्रोजन की आवश्यकता होती है।",
            "te": "అరటికి అధిక పొటాషియం, క్రమమైన కంపోస్ట్, విడతలవారీ నైట్రోజన్ అవసరం.",
        },
        "description": {
            "en": "Banana is a tropical fruit crop grown for local use and export.",
            "hi": "केला एक उष्णकटिबंधीय फल फसल है जिसे स्थानीय उपयोग और निर्यात के लिए उगाया जाता है।",
            "te": "అరటి ఒక ఉష్ణమండల పండు పంట; స్థానిక వినియోగం, ఎగుమతుల కోసం పండిస్తారు.",
        },
        "conditions": {
            "en": "Requires warm, humid climates and rich loamy soil.",
            "hi": "गर्म, आर्द्र जलवायु और उपजाऊ दोमट मिट्टी चाहिए।",
            "te": "వేడి, తేమ గల వాతావరణం మరియు సారవంతమైన లోమీ నేల అవసరం.",
        },
        "tips": {
            "en": "Irrigate regularly and keep potassium supply strong.",
            "hi": "नियमित सिंचाई करें और पोटाश की आपूर्ति बनाए रखें।",
            "te": "నియమితంగా నీరు పెట్టి పొటాషియం సరఫరా బలంగా ఉంచండి.",
        },
    },
    "mango": {
        "name": {"en": "Mango", "hi": "आम", "te": "మామిడి"},
        "fertilizer": {
            "en": "Apply age-based NPK with annual farmyard manure.",
            "hi": "पेड़ की आयु के अनुसार NPK दें और हर साल गोबर की खाद डालें।",
            "te": "చెట్టు వయస్సుకు అనుగుణంగా NPK ఇచ్చి, ప్రతి సంవత్సరం పశువుల ఎరువు వేయండి.",
        },
        "description": {
            "en": "Mango is a prized fruit known for its sweetness and aroma.",
            "hi": "आम अपनी मिठास और सुगंध के लिए प्रसिद्ध फल है।",
            "te": "మామిడి తన తీపి, సువాసనల కోసం ప్రసిద్ధి చెందిన పండు.",
        },
        "conditions": {
            "en": "Grows best in tropical or subtropical regions with good drainage.",
            "hi": "उष्णकटिबंधीय या उपोष्णकटिबंधीय क्षेत्रों में अच्छी जल निकासी के साथ बेहतर होता है।",
            "te": "ఉష్ణమండల లేదా ఉపఉష్ణమండల ప్రాంతాల్లో మంచి డ్రైనేజీతో ఉత్తమంగా పెరుగుతుంది.",
        },
        "tips": {
            "en": "Prune after harvest and protect trees from frost.",
            "hi": "कटाई के बाद छँटाई करें और पाले से बचाएँ।",
            "te": "కోత తర్వాత కత్తిరింపు చేసి మంచు నుంచి రక్షించండి.",
        },
    },
    "grapes": {
        "name": {"en": "Grapes", "hi": "अंगूर", "te": "ద్రాక్ష"},
        "fertilizer": {
            "en": "Use phosphorus and potassium-rich fertilizers; avoid excess nitrogen.",
            "hi": "फॉस्फोरस और पोटाश युक्त उर्वरक दें; अधिक नाइट्रोजन से बचें।",
            "te": "ఫాస్ఫరస్, పొటాషియం అధికంగా ఉన్న ఎరువులు వాడి అధిక నైట్రోజన్ నివారించండి.",
        },
        "description": {
            "en": "Grapes are grown for fresh fruit, juice, and processing.",
            "hi": "अंगूर ताजे फल, रस और प्रसंस्करण के लिए उगाए जाते हैं।",
            "te": "ద్రాక్షను తాజా పండ్లకు, రసానికి, ప్రాసెసింగ్‌కు పండిస్తారు.",
        },
        "conditions": {
            "en": "Needs warm, dry summers and well-drained soil.",
            "hi": "गर्म, शुष्क गर्मी और अच्छी जल निकासी वाली मिट्टी चाहिए।",
            "te": "వేడి, ఎండగా ఉండే వేసవి మరియు బాగా నీరు దిగే నేల అవసరం.",
        },
        "tips": {
            "en": "Train vines well and monitor pests like mealybugs.",
            "hi": "बेलों को सही प्रशिक्षण दें और मिलीबग जैसे कीटों पर नज़र रखें।",
            "te": "వెల్లలను సరిగా తీర్చిదిద్ది మీలీబగ్ వంటి పురుగులను గమనించండి.",
        },
    },
    "watermelon": {
        "name": {"en": "Watermelon", "hi": "तरबूज", "te": "పుచ్చకాయ"},
        "fertilizer": {
            "en": "Apply split NPK doses with strong potassium for sweetness.",
            "hi": "NPK को भागों में दें और मिठास के लिए पोटाश पर्याप्त रखें।",
            "te": "NPK ను విడతలుగా ఇచ్చి తీపి కోసం మంచి పొటాషియం అందించండి.",
        },
        "description": {
            "en": "Watermelon is a refreshing summer fruit rich in water.",
            "hi": "तरबूज पानी से भरपूर ताज़गी देने वाला ग्रीष्मकालीन फल है।",
            "te": "పుచ్చకాయ నీరు సమృద్ధిగా ఉన్న వేసవి కాలపు తేజోవంతమైన పండు.",
        },
        "conditions": {
            "en": "Best in sandy loam soils under warm temperatures.",
            "hi": "गर्म तापमान और बलुई दोमट मिट्टी में बेहतर होती है।",
            "te": "వేడి ఉష్ణోగ్రతలు, ఇసుక మిశ్రమ నేలల్లో బాగా పెరుగుతుంది.",
        },
        "tips": {
            "en": "Avoid over-irrigation and harvest when the tendril dries.",
            "hi": "अधिक सिंचाई न करें और बेल का पास वाला तंतु सूखने पर कटाई करें।",
            "te": "అధిక నీరు నివారించి దగ్గర టెండ్రిల్ ఎండినప్పుడు కోయండి.",
        },
    },
    "muskmelon": {
        "name": {"en": "Muskmelon", "hi": "खरबूजा", "te": "ఖర్బూజ"},
        "fertilizer": {
            "en": "Use split NPK doses and maintain potassium for fruit quality.",
            "hi": "NPK को भागों में दें और फल गुणवत्ता के लिए पोटाश बनाए रखें।",
            "te": "NPK ను విడతలుగా ఇచ్చి ఫల నాణ్యత కోసం పొటాషియం నిల్వ ఉంచండి.",
        },
        "description": {
            "en": "Muskmelon is a sweet and fragrant fruit crop.",
            "hi": "खरबूजा मीठा और सुगंधित फल है।",
            "te": "ఖర్బూజ తీపి, సువాసన గల పండు పంట.",
        },
        "conditions": {
            "en": "Prefers sandy loam soils and warm, dry conditions.",
            "hi": "बलुई दोमट मिट्टी और गर्म, शुष्क स्थिति पसंद है।",
            "te": "ఇసుక మిశ్రమ నేలలు, వేడి మరియు ఎండ పరిస్థితులు ఇష్టపడుతుంది.",
        },
        "tips": {
            "en": "Maintain spacing and avoid waterlogging.",
            "hi": "उचित दूरी रखें और जलभराव से बचें।",
            "te": "సరైన దూరం పాటించి నీరు నిల్వ కాకుండా చూడండి.",
        },
    },
    "apple": {
        "name": {"en": "Apple", "hi": "सेब", "te": "ఆపిల్"},
        "fertilizer": {
            "en": "Apply NPK in early spring and add compost in autumn.",
            "hi": "आरंभिक वसंत में NPK दें और शरद ऋतु में कम्पोस्ट डालें।",
            "te": "వసంత ప్రారంభంలో NPK ఇచ్చి, శరదృతువులో కంపోస్ట్ వేయండి.",
        },
        "description": {
            "en": "Apple is a temperate fruit rich in nutrients.",
            "hi": "सेब पोषक तत्वों से भरपूर समशीतोष्ण फल है।",
            "te": "ఆపిల్ పోషకాలు అధికంగా ఉండే సమశీతోష్ణ పండు.",
        },
        "conditions": {
            "en": "Needs cold winters and moderate summers.",
            "hi": "ठंडी सर्दियाँ और मध्यम गर्मियाँ चाहिए।",
            "te": "చల్లని శీతాకాలం, మితమైన వేసవి అవసరం.",
        },
        "tips": {
            "en": "Protect blossoms from frost and prune annually.",
            "hi": "फूलों को पाले से बचाएँ और हर साल छँटाई करें।",
            "te": "పూలను మంచు నుండి కాపాడి ప్రతి సంవత్సరం కత్తిరింపు చేయండి.",
        },
    },
    "orange": {
        "name": {"en": "Orange", "hi": "संतरा", "te": "నారింజ"},
        "fertilizer": {
            "en": "Use balanced NPK with extra potassium, zinc, and iron when needed.",
            "hi": "संतुलित NPK के साथ अतिरिक्त पोटाश, जिंक और आयरन दें।",
            "te": "సంతులిత NPK తో పాటు అవసరమైతే అదనపు పొటాషియం, జింక్, ఐరన్ ఇవ్వండి.",
        },
        "description": {
            "en": "Orange is a vitamin C-rich citrus fruit crop.",
            "hi": "संतरा विटामिन C से भरपूर खट्टा फल है।",
            "te": "నారింజ విటమిన్ C అధికంగా ఉన్న సిట్రస్ పండు పంట.",
        },
        "conditions": {
            "en": "Thrives in subtropical climates with well-drained soil.",
            "hi": "उपोष्णकटिबंधीय जलवायु और अच्छी जल निकासी वाली मिट्टी में अच्छी होती है।",
            "te": "ఉపఉష్ణమండల వాతావరణం, బాగా నీరు దిగే నేలలో బాగా పెరుగుతుంది.",
        },
        "tips": {
            "en": "Irrigate during dry spells and watch for citrus greening.",
            "hi": "सूखे समय सिंचाई करें और सिट्रस ग्रीनिंग पर नज़र रखें।",
            "te": "ఎండ సమయంలో నీరు పెట్టి సిట్రస్ గ్రీనింగ్ లక్షణాలను గమనించండి.",
        },
    },
    "papaya": {
        "name": {"en": "Papaya", "hi": "पपीता", "te": "బొప్పాయి"},
        "fertilizer": {
            "en": "Use higher nitrogen during growth and more potassium during fruiting.",
            "hi": "वृद्धि के समय अधिक नाइट्रोजन और फलन के समय अधिक पोटाश दें।",
            "te": "పెరుగుదల దశలో అధిక నైట్రోజన్, ఫల దశలో అధిక పొటాషియం ఇవ్వండి.",
        },
        "description": {
            "en": "Papaya is a nutritious tropical fruit crop.",
            "hi": "पपीता पौष्टिक उष्णकटिबंधीय फल फसल है।",
            "te": "బొప్పాయి పోషకవంతమైన ఉష్ణమండల పండు పంట.",
        },
        "conditions": {
            "en": "Prefers warm climates and sandy loam soils.",
            "hi": "गर्म जलवायु और बलुई दोमट मिट्टी पसंद है।",
            "te": "వేడి వాతావరణం, ఇసుక మిశ్రమ నేలలు ఇష్టపడుతుంది.",
        },
        "tips": {
            "en": "Ensure pollination balance and irrigate regularly.",
            "hi": "परागण संतुलन बनाए रखें और नियमित सिंचाई करें।",
            "te": "పరాగసంపర్కం సరిగా జరిగేలా చూసి క్రమంగా నీరు పెట్టండి.",
        },
    },
    "coconut": {
        "name": {"en": "Coconut", "hi": "नारियल", "te": "కొబ్బరి"},
        "fertilizer": {
            "en": "Apply annual NPK with magnesium and organic manure.",
            "hi": "हर साल NPK के साथ मैग्नीशियम और जैविक खाद दें।",
            "te": "ప్రతి సంవత్సరం NPK తో పాటు మాగ్నీషియం, సేంద్రీయ ఎరువు వేయండి.",
        },
        "description": {
            "en": "Coconut palms are grown for food, oil, and fiber.",
            "hi": "नारियल भोजन, तेल और रेशा के लिए उगाया जाता है।",
            "te": "కొబ్బరి ఆహారం, నూనె, రేశాల కోసం పండిస్తారు.",
        },
        "conditions": {
            "en": "Needs sandy coastal soils and high humidity.",
            "hi": "रेतीली तटीय मिट्टी और अधिक आर्द्रता चाहिए।",
            "te": "ఇసుక తీర నేలలు మరియు అధిక ఆర్ద్రత అవసరం.",
        },
        "tips": {
            "en": "Add organic manure regularly and monitor rhinoceros beetle.",
            "hi": "नियमित जैविक खाद दें और राइनोसीरस बीटल पर नज़र रखें।",
            "te": "క్రమం తప్పకుండా సేంద్రీయ ఎరువు వేసి రైనోసిరస్ బీటిల్‌ను గమనించండి.",
        },
    },
    "cotton": {
        "name": {"en": "Cotton", "hi": "कपास", "te": "పత్తి"},
        "fertilizer": {
            "en": "Use balanced NPK with extra potassium and organic manure.",
            "hi": "संतुलित NPK के साथ अतिरिक्त पोटाश और जैविक खाद दें।",
            "te": "సంతులిత NPK తో పాటు అదనపు పొటాషియం, సేంద్రీయ ఎరువు వాడండి.",
        },
        "description": {
            "en": "Cotton is a fiber crop used in the textile industry.",
            "hi": "कपास एक रेशा फसल है जिसका उपयोग वस्त्र उद्योग में होता है।",
            "te": "పత్తి వస్త్ర పరిశ్రమలో ఉపయోగించే తంతు పంట.",
        },
        "conditions": {
            "en": "Needs a long frost-free period and plenty of sunlight.",
            "hi": "लंबी पाला-मुक्त अवधि और भरपूर धूप चाहिए।",
            "te": "దీర్ఘకాలం మంచు లేకుండా, ఎక్కువ సూర్యకాంతి అవసరం.",
        },
        "tips": {
            "en": "Control bollworms and avoid waterlogging.",
            "hi": "बोलवर्म नियंत्रित करें और जलभराव से बचें।",
            "te": "బోల్‌వార్మ్ నియంత్రించి, నీరు నిల్వ కాకుండా చూడండి.",
        },
    },
    "jute": {
        "name": {"en": "Jute", "hi": "जूट", "te": "జూట్"},
        "fertilizer": {
            "en": "Apply nitrogen, phosphorus, and potash for stronger fiber.",
            "hi": "मजबूत रेशे के लिए नाइट्रोजन, फॉस्फोरस और पोटाश दें।",
            "te": "బలమైన ఫైబర్ కోసం నైట్రోజన్, ఫాస్ఫరస్, పొటాష్ ఇవ్వండి.",
        },
        "description": {
            "en": "Jute is a fiber crop used for ropes, bags, and sacks.",
            "hi": "जूट रस्सी, बैग और बोरे बनाने की रेशा फसल है।",
            "te": "జూట్ తాడులు, సంచులు, బస్తాల కోసం ఉపయోగించే తంతు పంట.",
        },
        "conditions": {
            "en": "Prefers warm, humid climates and alluvial soils.",
            "hi": "गर्म, आर्द्र जलवायु और जलोढ़ मिट्टी पसंद करता है।",
            "te": "వేడి, తేమ గల వాతావరణం మరియు అల్యూవియల్ నేలలు ఇష్టపడుతుంది.",
        },
        "tips": {
            "en": "Harvest at flowering stage for best fiber quality.",
            "hi": "श्रेष्ठ रेशा गुणवत्ता के लिए फूल आने की अवस्था में कटाई करें।",
            "te": "ఉత్తమ ఫైబర్ నాణ్యత కోసం పుష్పదశలో కోత కోయండి.",
        },
    },
    "coffee": {
        "name": {"en": "Coffee", "hi": "कॉफी", "te": "కాఫీ"},
        "fertilizer": {
            "en": "Use organic compost, potassium sulfate, and nitrogen after pruning.",
            "hi": "जैविक कम्पोस्ट, पोटाशियम सल्फेट और छँटाई के बाद नाइट्रोजन दें।",
            "te": "సేంద్రీయ కంపోస్ట్, పొటాషియం సల్ఫేట్, కత్తిరింపు తర్వాత నైట్రోజన్ ఇవ్వండి.",
        },
        "description": {
            "en": "Coffee is a beverage crop grown in tropical highlands.",
            "hi": "कॉफी उष्णकटिबंधीय उच्चभूमि में उगाई जाने वाली पेय फसल है।",
            "te": "కాఫీ ఉష్ణమండల ఎత్తైన ప్రాంతాల్లో పండించే పానీయ పంట.",
        },
        "conditions": {
            "en": "Requires shade, cool temperatures, and good drainage.",
            "hi": "इसे छाया, ठंडा तापमान और अच्छी जल निकासी चाहिए।",
            "te": "నీడ, చల్లటి ఉష్ణోగ్రత, మంచి డ్రైనేజీ అవసరం.",
        },
        "tips": {
            "en": "Control coffee berry borer and maintain shade trees.",
            "hi": "कॉफी बेरी बोरर नियंत्रित करें और छायादार पेड़ बनाए रखें।",
            "te": "కాఫీ బెర్రీ బోరర్‌ను నియంత్రించి, నీడిచ్చే చెట్లను నిల్వ ఉంచండి.",
        },
    },
}

LABEL_MAPPING = {
    0: "rice",
    1: "wheat",
    2: "maize",
    3: "chickpea",
    4: "kidneybeans",
    5: "pigeonpeas",
    6: "mothbeans",
    7: "mungbean",
    8: "blackgram",
    9: "lentil",
    10: "pomegranate",
    11: "banana",
    12: "mango",
    13: "grapes",
    14: "watermelon",
    15: "muskmelon",
    16: "apple",
    17: "orange",
    18: "papaya",
    19: "coconut",
    20: "cotton",
    21: "jute",
    22: "coffee",
}


st.set_page_config(page_title="DeepCropCare", layout="wide")


def t(key, lang):
    return LANGUAGE_LABELS[lang].get(key, LANGUAGE_LABELS["en"].get(key, key))


def get_disease_meta(class_name):
    metadata_key = DISEASE_METADATA_ALIASES.get(class_name, class_name)
    return DISEASE_METADATA.get(metadata_key, DEFAULT_DISEASE_FALLBACK.get(metadata_key, {
        "display": {"en": class_name.replace("___", " ").replace("__", " ").replace("_", " ")},
        "advice": {"en": "Follow integrated crop management practices."},
    }))


def disease_display(class_name, lang):
    meta = get_disease_meta(class_name)
    return meta["display"].get(lang) or meta["display"].get("en") or class_name


def disease_advice(class_name, lang):
    meta = get_disease_meta(class_name)
    return meta["advice"].get(lang) or meta["advice"].get("en") or t("na", lang)


def normalize_predicted_class(class_name):
    swap_map = {
        "Wheat_leaf_stripe_rust": "Wheat_leaf_septoria",
        "Wheat_leaf_septoria": "Wheat_leaf_stripe_rust",
        "Wheatleaf_septoria": "Wheat_leaf_stripe_rust",
    }
    return swap_map.get(class_name, class_name)


def crop_text(crop_key, field, lang):
    return CROP_METADATA[crop_key][field].get(lang) or CROP_METADATA[crop_key][field]["en"]


def build_assistant_icon_data_uri():
    svg = """
    <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 160 160'>
      <defs>
        <linearGradient id='g1' x1='0' x2='1'>
          <stop offset='0%' stop-color='#ff8a3d'/>
          <stop offset='100%' stop-color='#d84c1b'/>
        </linearGradient>
      </defs>
      <rect x='40' y='18' rx='18' ry='18' width='56' height='44' fill='url(#g1)' stroke='#a53b17' stroke-width='3'/>
      <rect x='48' y='28' rx='8' ry='8' width='40' height='26' fill='#242424'/>
      <rect x='56' y='34' width='7' height='8' fill='#ffe65e'/>
      <rect x='73' y='34' width='7' height='8' fill='#ffe65e'/>
      <rect x='63' y='46' width='12' height='4' rx='2' fill='#ffe65e'/>
      <circle cx='38' cy='38' r='6' fill='#ffd84d'/>
      <circle cx='101' cy='38' r='6' fill='#ffd84d'/>
      <rect x='54' y='64' rx='10' ry='10' width='28' height='28' fill='url(#g1)' stroke='#a53b17' stroke-width='3'/>
      <line x1='54' y1='74' x2='34' y2='88' stroke='#3c3c3c' stroke-width='6' stroke-linecap='round'/>
      <line x1='82' y1='74' x2='102' y2='88' stroke='#3c3c3c' stroke-width='6' stroke-linecap='round'/>
      <line x1='63' y1='92' x2='55' y2='118' stroke='#3c3c3c' stroke-width='6' stroke-linecap='round'/>
      <line x1='74' y1='92' x2='84' y2='118' stroke='#3c3c3c' stroke-width='6' stroke-linecap='round'/>
      <circle cx='29' cy='92' r='4' fill='#3c3c3c'/>
      <path d='M15 80 h18 a10 10 0 0 1 10 10 v18 a10 10 0 0 1 -10 10 h-18 z' fill='#f39a1d' stroke='#bd6d0a' stroke-width='3'/>
      <path d='M33 82 q12 0 16 8' fill='none' stroke='#f7c14b' stroke-width='4' stroke-linecap='round'/>
      <path d='M22 95 q7 -6 14 0' fill='none' stroke='#49a7ff' stroke-width='5' stroke-linecap='round'/>
      <circle cx='22' cy='110' r='2.5' fill='#49a7ff'/>
      <circle cx='26' cy='116' r='2.5' fill='#49a7ff'/>
    </svg>
    """
    return f"data:image/svg+xml;utf8,{quote(svg)}"


ASSISTANT_ICON = build_assistant_icon_data_uri()


def build_disease_prompt(class_name, lang):
    return t("assistant_auto_prompt", lang).format(disease=disease_display(class_name, lang))


def build_fallback_disease_report(class_name, lang):
    disease_name = disease_display(class_name, lang)
    advice = disease_advice(class_name, lang)
    return "\n\n".join(
        [
            f"### {t('assistant_fallback_title', lang)}",
            t("assistant_fallback_description", lang).format(disease=disease_name),
            f"**{t('label_symptoms', lang)}:** {t('assistant_fallback_symptoms', lang)}",
            f"**{t('label_cure', lang)}:** {t('assistant_fallback_cure', lang)}",
            f"**{t('label_prevention', lang)}:** {t('assistant_fallback_prevention', lang)}",
            f"**{t('label_fertilizer_care', lang)}:** {t('assistant_fallback_fertilizer', lang).format(advice=advice)}",
        ]
    )


def build_disease_report_text(class_name, confidence, lang, image_name, heatmap_available):
    lines = [
        "DeepCropCare",
        t("report_summary", lang),
        "",
        f"{t('report_generated_on', lang)}: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"{t('report_image_name', lang)}: {image_name or t('na', lang)}",
        f"{t('report_detected_disease', lang)}: {disease_display(class_name, lang)}",
        f"{t('confidence', lang)}: {confidence:.2f}%",
        f"{t('recommended_action', lang)}: {disease_advice(class_name, lang)}",
        f"{t('report_heatmap_status', lang)}: {t('report_heatmap_ready', lang) if heatmap_available else t('report_heatmap_unavailable', lang)}",
        "",
        build_fallback_disease_report(class_name, lang),
    ]
    return "\n".join(lines)


def build_crop_report_text(crop_key, lang, inputs):
    crop_name = crop_text(crop_key, "name", lang)
    lines = [
        "DeepCropCare",
        t("crop_report_heading", lang),
        "",
        f"{t('report_generated_on', lang)}: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"{t('report_crop_name', lang)}: {crop_name}",
        f"{t('report_input_summary', lang)}:",
        f"{t('nitrogen', lang)}: {inputs['nitrogen']}",
        f"{t('phosphorus', lang)}: {inputs['phosphorus']}",
        f"{t('potassium', lang)}: {inputs['potassium']}",
        f"{t('soil_ph', lang)}: {inputs['ph']}",
        f"{t('rainfall', lang)}: {inputs['rain']}",
        f"{t('report_weather_used', lang)}: {inputs['city']}",
        f"{t('report_temperature', lang)}: {inputs['temp']}",
        f"{t('humidity', lang)}: {inputs['hum']}",
        "",
        f"{t('report_crop_description', lang)}: {crop_text(crop_key, 'description', lang)}",
        f"{t('report_crop_conditions', lang)}: {crop_text(crop_key, 'conditions', lang)}",
        f"{t('report_crop_fertilizer', lang)}: {crop_text(crop_key, 'fertilizer', lang)}",
        f"{t('report_crop_tip', lang)}: {crop_text(crop_key, 'tips', lang)}",
    ]
    return "\n".join(lines)


def _pick_pdf_font_path(lang):
    app_dir = Path(__file__).resolve().parent
    bundled_font_dir = app_dir / "fonts"
    candidates = []
    if lang == "hi":
        candidates.extend(
            [
                bundled_font_dir / "NotoSansDevanagari-Regular.ttf",
                bundled_font_dir / "NotoSerifDevanagari-Regular.ttf",
                "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
                "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Regular.ttf",
                "/usr/share/fonts/opentype/noto/NotoSerifDevanagari-Regular.ttf",
            ]
        )
    elif lang == "te":
        candidates.extend(
            [
                bundled_font_dir / "NotoSerifTelugu-Regular-static.ttf",
                bundled_font_dir / "Lohit-Telugu.ttf",
                bundled_font_dir / "NotoSansTelugu-Regular.ttf",
                bundled_font_dir / "NotoSerifTelugu-Regular.ttf",
                "/System/Library/Fonts/Supplemental/Telugu Sangam MN.ttc",
                "/System/Library/Fonts/Supplemental/Telugu MN.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansTelugu-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSerifTelugu-Regular.ttf",
                "/usr/share/fonts/opentype/noto/NotoSerifTelugu-Regular.ttf",
            ]
        )
    candidates.extend(
        [
            bundled_font_dir / "ArialUnicode.ttf",
            bundled_font_dir / "Arial Unicode.ttf",
            bundled_font_dir / "NotoSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    candidates.extend(
        [
            bundled_font_dir / "Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            return str(path)
    return None


def _validate_pdf_font_path(font_path, lang):
    if not font_path:
        return None
    return font_path


def _ensure_static_pdf_font(font_path):
    if not font_path:
        return None

    font_path = Path(font_path)
    try:
        font = TTFont(str(font_path))
    except Exception:
        return str(font_path)

    if "fvar" not in font and "gvar" not in font:
        return str(font_path)

    temp_dir = Path(tempfile.gettempdir()) / "deepcropcare-fonts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    font_hash = hashlib.sha1(str(font_path).encode("utf-8")).hexdigest()[:12]
    static_path = temp_dir / f"{font_path.stem}-{font_hash}-static.ttf"
    if static_path.exists():
        return str(static_path)

    axes = {axis.axisTag: axis.defaultValue for axis in font["fvar"].axes}
    static_font = instantiateVariableFont(font, axes, inplace=False)
    static_font.save(str(static_path))
    return str(static_path)


def _get_pdf_font(lang, size):
    font_path = _pick_pdf_font_path(lang)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap_pdf_line(draw, text, font, max_width):
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _normalize_pdf_text_line(text):
    line = str(text or "").strip()
    if not line:
        return ""
    if line.startswith("### "):
        return line[4:].strip().upper()
    line = line.replace("**", "")
    return line


def _pdf_escape_text(text):
    safe = str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return safe.encode("latin-1", "replace").decode("latin-1")


def _build_raw_pdf(page_specs):
    objects = []

    def add_object(data):
        objects.append(data)
        return len(objects)

    font_times = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>")
    font_helvetica = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_helvetica_bold = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_object_ids = []

    for page in page_specs:
        image_entries = []
        xobject_lines = []
        for index, image in enumerate(page.get("images", []), start=1):
            image_obj_id = add_object(
                b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n"
                % (image["width"], image["height"], len(image["data"]))
                + image["data"]
                + b"\nendstream"
            )
            image_name = f"/Im{index}"
            image_entries.append((image_name, image_obj_id, image))
            xobject_lines.append(f"{image_name} {image_obj_id} 0 R")

        content_parts = ["BT"]
        for text_item in page.get("texts", []):
            font_key = {
                "times": "/F1",
                "bold": "/F3",
            }.get(text_item["font"], "/F2")
            r, g, b = text_item.get("color", (0, 0, 0))
            content_parts.append(
                f"{font_key} {text_item['size']} Tf {r:.4f} {g:.4f} {b:.4f} rg "
                f"1 0 0 1 {text_item['x']:.2f} {text_item['y']:.2f} Tm "
                f"({_pdf_escape_text(text_item['text'])}) Tj"
            )
        content_parts.append("ET")

        for image_name, _, image in image_entries:
            content_parts.append(
                "q "
                f"{image['draw_width']:.2f} 0 0 {image['draw_height']:.2f} "
                f"{image['x']:.2f} {image['y']:.2f} cm {image_name} Do Q"
            )

        content_stream = "\n".join(content_parts).encode("latin-1", "replace")
        content_obj_id = add_object(
            b"<< /Length %d >>\nstream\n" % len(content_stream)
            + content_stream
            + b"\nendstream"
        )

        resources = (
            f"<< /Font << /F1 {font_times} 0 R /F2 {font_helvetica} 0 R /F3 {font_helvetica_bold} 0 R >>"
        )
        if xobject_lines:
            resources += f" /XObject << {' '.join(xobject_lines)} >>"
        resources += " >>"

        page_obj_id = add_object(
            f"<< /Type /Page /Parent PAGES_REF /MediaBox [0 0 612 792] "
            f"/Resources {resources} /Contents {content_obj_id} 0 R >>"
        )
        page_object_ids.append(page_obj_id)

    pages_obj_id = add_object(
        f"<< /Type /Pages /Count {len(page_object_ids)} /Kids [{' '.join(f'{obj} 0 R' for obj in page_object_ids)}] >>"
    )
    catalog_obj_id = add_object(f"<< /Type /Catalog /Pages {pages_obj_id} 0 R >>")

    serialized = []
    for obj in objects:
        if isinstance(obj, bytes):
            serialized.append(obj.replace(b"PAGES_REF", f"{pages_obj_id} 0 R".encode()))
        else:
            serialized.append(obj.replace("PAGES_REF", f"{pages_obj_id} 0 R").encode("latin-1", "replace"))

    buffer = BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(serialized, start=1):
        offsets.append(buffer.tell())
        buffer.write(f"{index} 0 obj\n".encode("latin-1"))
        buffer.write(obj)
        buffer.write(b"\nendobj\n")

    xref_start = buffer.tell()
    buffer.write(f"xref\n0 {len(serialized) + 1}\n".encode("latin-1"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
    buffer.write(
        (
            f"trailer\n<< /Size {len(serialized) + 1} /Root {catalog_obj_id} 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        ).encode("latin-1")
    )
    return buffer.getvalue()


def _build_vector_pdf_report_bytes(title, body_text, lang, image_blocks=None):
    page_width, page_height = 612, 792
    left_margin = 72
    right_margin = 72
    max_text_width = page_width - left_margin - right_margin
    wrap_draw = ImageDraw.Draw(Image.new("RGB", (page_width, page_height), "white"))
    title_font = _get_pdf_font(lang, 20)
    subtitle_font = _get_pdf_font(lang, 20)
    body_font = _get_pdf_font(lang, 12)
    heading_font = _get_pdf_font(lang, 14)

    raw_lines = [_normalize_pdf_text_line(line) for line in body_text.splitlines()]
    lines = [line for line in raw_lines if line or line == ""]

    page_specs = [{"texts": [], "images": []}]

    def add_text(page_index, text, x, top_y, size, font, color=(0, 0, 0)):
        page_specs[page_index]["texts"].append(
            {
                "text": text,
                "x": x,
                "y": page_height - top_y,
                "size": size,
                "font": font,
                "color": color,
            }
        )

    first_line = lines[0] if lines else "DeepCropCare"
    second_line = next((line for line in lines[1:] if line), title)

    first_width = wrap_draw.textlength(first_line, font=title_font)
    second_width = wrap_draw.textlength(second_line, font=subtitle_font)
    add_text(0, first_line, max(72, (page_width - first_width) / 2), 103, 20, "times")
    add_text(0, second_line, max(72, (page_width - second_width) / 2), 142, 20, "times")

    y = 212
    body_start_index = 2
    for index, raw_line in enumerate(lines[body_start_index:], start=body_start_index):
        if not raw_line:
            y += 16
            continue

        is_primary_heading = raw_line.isupper()
        color = (0, 0, 0)
        font_name = "helvetica"
        font_size = 12

        if is_primary_heading:
            font_name = "bold"
            font_size = 14
            color = (0.0588, 0.278, 0.38)
            text_to_draw = raw_line.title()
        else:
            text_to_draw = raw_line
            if raw_line.startswith("Detected Disease:"):
                color = (1, 0, 0)

        wrap_font = heading_font if font_size == 14 else body_font
        wrapped_lines = _wrap_pdf_line(wrap_draw, text_to_draw, wrap_font, max_text_width)

        if y + (len(wrapped_lines) * 18) > 720:
            page_specs.append({"texts": [], "images": []})
            current_page = len(page_specs) - 1
            y = 90
        else:
            current_page = len(page_specs) - 1

        for wrapped_line in wrapped_lines:
            add_text(current_page, wrapped_line, left_margin, y, font_size, font_name, color)
            y += 29 if font_size == 14 else 24

    if image_blocks:
        for start in range(0, len(image_blocks), 2):
            image_page = {"texts": [], "images": []}
            title_text = "AI Heatmap Images"
            title_width = wrap_draw.textlength(title_text, font=title_font)
            image_page["texts"].append(
                {
                    "text": title_text,
                    "x": max(72, (page_width - title_width) / 2),
                    "y": page_height - 103,
                    "size": 20,
                    "font": "times",
                    "color": (0, 0, 0),
                }
            )

            slots = [
                {"label_top": 150, "box": (72, 170, 540, 360)},
                {"label_top": 420, "box": (72, 440, 540, 700)},
            ]

            for (label, image_obj), slot in zip(image_blocks[start : start + 2], slots):
                label_text = _normalize_pdf_text_line(label).title()
                image_page["texts"].append(
                    {
                        "text": label_text,
                        "x": 72,
                        "y": page_height - slot["label_top"],
                        "size": 14,
                        "font": "bold",
                        "color": (0, 0, 0),
                    }
                )

                image_rgb = image_obj.convert("RGB")
                box_x1, box_y1, box_x2, box_y2 = slot["box"]
                max_width = box_x2 - box_x1
                max_height = box_y2 - box_y1
                scale = min(max_width / image_rgb.width, max_height / image_rgb.height)
                draw_width = image_rgb.width * scale
                draw_height = image_rgb.height * scale
                x = box_x1 + ((max_width - draw_width) / 2)
                top_y = box_y1 + ((max_height - draw_height) / 2)

                image_buffer = BytesIO()
                image_rgb.save(image_buffer, format="JPEG", quality=92)
                image_page["images"].append(
                    {
                        "data": image_buffer.getvalue(),
                        "width": image_rgb.width,
                        "height": image_rgb.height,
                        "x": x,
                        "y": page_height - top_y - draw_height,
                        "draw_width": draw_width,
                        "draw_height": draw_height,
                    }
                )

            page_specs.append(image_page)

    return _build_raw_pdf(page_specs)


def _draw_rounded_image(canvas, image_obj, box, radius=32):
    x1, y1, x2, y2 = box
    image_copy = image_obj.convert("RGB")
    image_copy.thumbnail((x2 - x1, y2 - y1))
    paste_x = x1 + ((x2 - x1) - image_copy.width) // 2
    paste_y = y1 + ((y2 - y1) - image_copy.height) // 2

    mask = Image.new("L", (image_copy.width, image_copy.height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, image_copy.width, image_copy.height), radius=radius, fill=255)
    canvas.paste(image_copy, (paste_x, paste_y), mask)


def _configure_pdf_font(pdf, lang):
    if lang == "en":
        return "Times"

    font_path = _validate_pdf_font_path(_pick_pdf_font_path(lang), lang)
    family = f"DeepCropCare-{lang}"
    if not font_path:
        raise RuntimeError(
            f"Missing Unicode font for language '{lang}'. Add a compatible font file under "
            f"'{Path(__file__).resolve().parent / 'fonts'}'."
        )
    pdf.add_font(family, fname=_ensure_static_pdf_font(font_path))
    pdf.set_text_shaping(True)
    return family


def build_pdf_report_bytes(title, body_text, lang, image_blocks=None):
    page_width, page_height = 612, 792
    left_margin = 72
    right_margin = 72
    max_text_width = page_width - left_margin - right_margin
    raw_lines = [_normalize_pdf_text_line(line) for line in body_text.splitlines()]
    lines = [line for line in raw_lines if line or line == ""]
    pdf = FPDF(unit="pt", format="Letter")
    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(left_margin, 0, right_margin)
    pdf.add_page()
    font_family = _configure_pdf_font(pdf, lang)

    def write_line(text, y, size, color=(0, 0, 0), align="L"):
        pdf.set_text_color(*color)
        pdf.set_font(font_family, size=size)
        pdf.set_xy(left_margin, y)
        pdf.multi_cell(max_text_width, size + 6, text, align=align)

    first_line = lines[0] if lines else "DeepCropCare"
    second_line = next((line for line in lines[1:] if line), title)
    write_line(first_line, 103, 20, align="C")
    second_color = (18, 122, 63) if second_line.startswith(t("report_crop_name", lang) + ":") else (0, 0, 0)
    write_line(second_line, 142, 20, color=second_color, align="C")
    y = 212

    for raw_line in lines[2:]:
        if not raw_line:
            y += 16
            continue

        is_heading = (
            raw_line == t("assistant_fallback_title", lang)
            or raw_line.isupper()
            or (raw_line.endswith(":") and len(raw_line) < 55)
        )
        fill = (15, 71, 97) if is_heading else (0, 0, 0)
        text_to_draw = raw_line.title() if is_heading and raw_line.isascii() else raw_line
        if not is_heading and raw_line.startswith(t("report_detected_disease", lang) + ":"):
            fill = (255, 0, 0)
        if not is_heading and raw_line.startswith(t("report_crop_name", lang) + ":"):
            fill = (18, 122, 63)
        font_size = 14 if is_heading else 12
        line_step = 29 if is_heading else 24
        pdf.set_font(font_family, size=font_size)
        required_height = max(line_step, pdf.multi_cell(max_text_width, line_step, text_to_draw, dry_run=True, output="HEIGHT"))
        if y + required_height > 720:
            pdf.add_page()
            y = 90
        pdf.set_text_color(*fill)
        pdf.set_font(font_family, size=font_size)
        pdf.set_xy(left_margin, y)
        pdf.multi_cell(max_text_width, line_step, text_to_draw)
        y = pdf.get_y()

    if image_blocks:
        for start in range(0, len(image_blocks), 2):
            pdf.add_page()
            heatmap_title = t("heatmap_images_title", lang)
            write_line(heatmap_title, 103, 20, align="C")

            chunk = image_blocks[start : start + 2]
            slots = [
                (72, 170, 540, 360),
                (72, 440, 540, 700),
            ]

            for (label, image_obj), box in zip(chunk, slots):
                normalized_label = _normalize_pdf_text_line(label)
                label_text = normalized_label.title() if normalized_label.isascii() else normalized_label
                pdf.set_text_color(0, 0, 0)
                pdf.set_font(font_family, size=14)
                pdf.set_xy(box[0], box[1] - 24)
                pdf.multi_cell(box[2] - box[0], 18, label_text)

                image_rgb = image_obj.convert("RGB")
                box_x1, box_y1, box_x2, box_y2 = box
                max_width = box_x2 - box_x1
                max_height = box_y2 - box_y1
                scale = min(max_width / image_rgb.width, max_height / image_rgb.height)
                draw_width = image_rgb.width * scale
                draw_height = image_rgb.height * scale
                x = box_x1 + ((max_width - draw_width) / 2)
                y_img = box_y1 + ((max_height - draw_height) / 2)

                image_buffer = BytesIO()
                image_rgb.save(image_buffer, format="PNG")
                image_buffer.seek(0)
                pdf.image(image_buffer, x=x, y=y_img, w=draw_width, h=draw_height)

    return bytes(pdf.output())


def inject_tab_switch(tab_text):
    safe_text = tab_text.replace("\\", "\\\\").replace("'", "\\'")
    components.html(
        f"""
        <script>
        const targetText = '{safe_text}';
        let attempts = 0;
        const timer = setInterval(() => {{
          attempts += 1;
          const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
          const target = Array.from(tabs).find((tab) => tab.innerText.includes(targetText));
          if (target) {{
            target.click();
            clearInterval(timer);
          }}
          if (attempts > 30) {{
            clearInterval(timer);
          }}
        }}, 150);
        </script>
        """,
        height=0,
    )


def inject_helper_icon(icon_src, hint_text, note_text, trigger_label, loading_text, visible_tab_text):
    safe_hint = hint_text.replace("\\", "\\\\").replace("'", "\\'")
    safe_note = note_text.replace("\\", "\\\\").replace("'", "\\'")
    safe_icon = icon_src.replace("\\", "\\\\").replace("'", "\\'")
    safe_trigger = trigger_label.replace("\\", "\\\\").replace("'", "\\'")
    safe_loading = loading_text.replace("\\", "\\\\").replace("'", "\\'")
    safe_visible_tab = visible_tab_text.replace("\\", "\\\\").replace("'", "\\'")
    components.html(
        f"""
        <script>
        const parentDoc = window.parent.document;
        const parentWin = window.parent;
        const oldNode = parentDoc.getElementById('deepcropcare-helper-dock');
        if (oldNode) oldNode.remove();
        const oldOverlay = parentDoc.getElementById('deepcropcare-helper-overlay');
        if (oldOverlay) oldOverlay.remove();
        if (parentWin.deepCropCareHelperTimer) {{
          clearInterval(parentWin.deepCropCareHelperTimer);
          parentWin.deepCropCareHelperTimer = null;
        }}

        const helperButton = Array.from(parentDoc.querySelectorAll('button')).find(
          (button) => button.innerText.trim() === '{safe_trigger}'
        );
        if (helperButton) {{
          const wrapper = helperButton.closest('div[data-testid="stButton"]');
          if (wrapper) {{
            wrapper.style.position = 'fixed';
            wrapper.style.left = '-9999px';
            wrapper.style.opacity = '0';
            wrapper.style.pointerEvents = 'none';
            wrapper.style.width = '1px';
            wrapper.style.height = '1px';
            wrapper.style.overflow = 'hidden';
          }}
        }}

        const dock = parentDoc.createElement('div');
        dock.id = 'deepcropcare-helper-dock';
        dock.innerHTML = `
          <style>
            #deepcropcare-helper-dock {{
              position: fixed;
              right: 26px;
              bottom: 26px;
              z-index: 9999;
              display: flex;
              flex-direction: column;
              align-items: center;
              gap: 10px;
              cursor: pointer;
              user-select: none;
            }}
            #deepcropcare-helper-dock .helper-orb {{
              width: 74px;
              height: 74px;
              border-radius: 999px;
              display: flex;
              align-items: center;
              justify-content: center;
              background: radial-gradient(circle at 30% 30%, rgba(255,177,66,.96), rgba(211,78,28,.96));
              box-shadow: 0 14px 28px rgba(0,0,0,.32), 0 0 0 8px rgba(255,172,64,.09);
              border: 3px solid rgba(255,255,255,.18);
              animation: agribot-float 2.1s ease-in-out infinite;
            }}
            #deepcropcare-helper-dock .helper-orb img {{
              width: 54px;
              height: 54px;
              object-fit: contain;
              border-radius: 50%;
              pointer-events: none;
            }}
            #deepcropcare-helper-dock .helper-caption {{
              max-width: 180px;
              text-align: center;
              color: #dfe8d9;
              font-size: 13px;
              line-height: 1.45;
              text-shadow: 0 2px 10px rgba(0,0,0,.35);
            }}
            #deepcropcare-helper-overlay {{
              position: fixed;
              inset: 0;
              display: flex;
              align-items: center;
              justify-content: center;
              background: rgba(8, 14, 12, 0.45);
              z-index: 10000;
              backdrop-filter: blur(2px);
            }}
            #deepcropcare-helper-overlay .overlay-card {{
              display: flex;
              align-items: center;
              gap: 12px;
              padding: 14px 18px;
              border-radius: 14px;
              background: rgba(16, 24, 20, 0.96);
              color: #eef6ee;
              box-shadow: 0 18px 38px rgba(0,0,0,.35);
              border: 1px solid rgba(255,255,255,.1);
              font-size: 16px;
              font-weight: 600;
            }}
            #deepcropcare-helper-overlay .overlay-spinner {{
              width: 18px;
              height: 18px;
              border-radius: 50%;
              border: 3px solid rgba(255,255,255,.22);
              border-top-color: #31c555;
              animation: helper-spin 0.9s linear infinite;
            }}
            @keyframes agribot-float {{
              0% {{ transform: translateY(0px) scale(1); }}
              50% {{ transform: translateY(-14px) scale(1.03); }}
              100% {{ transform: translateY(0px) scale(1); }}
            }}
            @keyframes helper-spin {{
              to {{ transform: rotate(360deg); }}
            }}
          </style>
          <div class="helper-orb" title="{safe_note}">
            <img src="{safe_icon}" alt="{safe_note}" />
          </div>
          <div class="helper-caption">{safe_hint}</div>
        `;

        dock.addEventListener('click', () => {{
          if (!helperButton) return;
          const existingOverlay = parentDoc.getElementById('deepcropcare-helper-overlay');
          if (existingOverlay) existingOverlay.remove();
          const overlay = parentDoc.createElement('div');
          overlay.id = 'deepcropcare-helper-overlay';
          overlay.innerHTML = `
            <div class="overlay-card">
              <div class="overlay-spinner"></div>
              <div>{safe_loading}</div>
            </div>
          `;
          parentDoc.body.appendChild(overlay);
          window.setTimeout(() => {{
            const staleOverlay = parentDoc.getElementById('deepcropcare-helper-overlay');
            if (staleOverlay) staleOverlay.remove();
          }}, 8000);
          helperButton.click();
        }});

        parentDoc.body.appendChild(dock);

        const syncDockVisibility = () => {{
          const activeTab = Array.from(parentDoc.querySelectorAll('button[role="tab"][aria-selected="true"]'))[0];
          const activeText = activeTab ? activeTab.innerText.replace(/\\s+/g, ' ').trim() : '';
          dock.style.display = activeText.includes('{safe_visible_tab}') ? 'flex' : 'none';
        }};

        syncDockVisibility();
        parentWin.deepCropCareHelperTimer = setInterval(syncDockVisibility, 200);
        </script>
        """,
        height=0,
    )


def remove_helper_icon():
    components.html(
        """
        <script>
        const parentWin = window.parent;
        if (parentWin.deepCropCareHelperTimer) {
            clearInterval(parentWin.deepCropCareHelperTimer);
            parentWin.deepCropCareHelperTimer = null;
        }
        const oldNode = window.parent.document.getElementById('deepcropcare-helper-dock');
        if (oldNode) oldNode.remove();
        const oldOverlay = window.parent.document.getElementById('deepcropcare-helper-overlay');
        if (oldOverlay) oldOverlay.remove();
        </script>
        """,
        height=0,
    )


def get_gemini_backend():
    global legacy_genai
    if modern_genai is not None:
        return "modern", modern_genai
    if legacy_genai is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*google.generativeai.*",
                category=FutureWarning,
            )
            import google.generativeai as imported_legacy_genai

        legacy_genai = imported_legacy_genai
    return "legacy", legacy_genai


def build_chat_session(api_key, model_id, system_instruction):
    backend_name, backend = get_gemini_backend()
    if backend_name == "modern":
        return {
            "backend": "modern",
            "client": backend.Client(api_key=api_key),
            "model_id": model_id,
            "system_instruction": system_instruction,
            "history": [],
        }

    backend.configure(api_key=api_key)
    model = backend.GenerativeModel(model_name=model_id, system_instruction=system_instruction)
    return {
        "backend": "legacy",
        "chat": model.start_chat(history=[]),
    }


def chat_session_send_message(chat_session, prompt):
    if chat_session.get("backend") == "modern":
        history = chat_session.setdefault("history", [])
        transcript_lines = [f"System: {chat_session['system_instruction']}"]
        for message in history:
            role = "Assistant" if message["role"] == "assistant" else "User"
            transcript_lines.append(f"{role}: {message['content']}")
        transcript_lines.append(f"User: {prompt}")
        transcript_lines.append("Assistant:")
        response = chat_session["client"].models.generate_content(
            model=chat_session["model_id"],
            contents="\n".join(transcript_lines),
        )
        response_text = getattr(response, "text", "") or ""
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": response_text})
        return response_text

    response = chat_session["chat"].send_message(prompt)
    return response.text


def inject_feature_launcher_feedback(active_label, launcher_labels):
    active_safe = active_label.replace("\\", "\\\\").replace("'", "\\'")
    labels_json = json.dumps(list(launcher_labels))
    components.html(
        f"""
        <script>
        const parentDoc = window.parent.document;
        const labels = {labels_json};
        const activeLabel = '{active_safe}';

        const findFeatureButtons = () => Array.from(parentDoc.querySelectorAll('button')).filter((button) => {{
          const text = button.innerText.replace(/\\s+/g, ' ').trim();
          return labels.some((label) => text === label);
        }});

        const applyState = (selectedText) => {{
          const buttons = findFeatureButtons();
          buttons.forEach((button) => {{
            const text = button.innerText.replace(/\\s+/g, ' ').trim();
            const isActive = text === selectedText;
            button.style.transition = 'transform 140ms ease-out, background 140ms ease-out, border-color 140ms ease-out, box-shadow 140ms ease-out';
            if (isActive) {{
              button.style.background = 'linear-gradient(180deg, rgba(132,255,192,0.22), rgba(18,69,48,0.34))';
              button.style.border = '1px solid rgba(132,255,192,0.72)';
              button.style.boxShadow = 'inset 0 10px 24px rgba(255,255,255,0.04), inset 0 -16px 28px rgba(0,0,0,0.16), 0 0 0 1px rgba(132,255,192,0.2), 0 12px 26px rgba(8,31,24,0.22), 0 0 24px rgba(118,255,184,0.16)';
              button.style.transform = 'scale(0.96)';
              button.style.color = '#ffffff';
            }} else {{
              button.style.background = 'rgba(255,255,255,0.05)';
              button.style.border = '1px solid rgba(255,255,255,0.12)';
              button.style.boxShadow = '0 10px 28px rgba(5,12,28,0.12)';
              button.style.transform = 'scale(1)';
              button.style.color = '#eef6ee';
            }}
            if (!button.dataset.featureBound) {{
              button.dataset.featureBound = '1';
              button.addEventListener('click', () => applyState(text));
            }}
          }});
        }};

        let attempts = 0;
        const timer = setInterval(() => {{
          attempts += 1;
          const buttons = findFeatureButtons();
          if (buttons.length >= labels.length) {{
            applyState(activeLabel);
            clearInterval(timer);
          }}
          if (attempts > 30) {{
            clearInterval(timer);
          }}
        }}, 120);
        </script>
        """,
        height=0,
    )


def inject_input_theme():
    components.html(
        """
        <script>
        const parentDoc = window.parent.document;
        const applyFieldTheme = () => {
          const isLightMode = window.parent.matchMedia && window.parent.matchMedia('(prefers-color-scheme: light)').matches;
          const textColor = isLightMode ? '#111111' : '#eef6ee';
          const shellSelectors = [
            'div[data-testid="stNumberInput"] [data-baseweb="base-input"]',
            'div[data-testid="stTextInput"] [data-baseweb="base-input"]',
            'div[data-testid="stSelectbox"] [data-baseweb="select"] > div'
          ];
          shellSelectors.forEach((selector) => {
            parentDoc.querySelectorAll(selector).forEach((node) => {
              node.style.background = 'rgba(255, 255, 255, 0.10)';
              node.style.border = '1px solid rgba(255, 255, 255, 0.14)';
              node.style.borderRadius = '12px';
              node.style.boxShadow = 'none';
            });
          });

          parentDoc.querySelectorAll('div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input').forEach((node) => {
            node.style.background = 'transparent';
            node.style.backgroundColor = 'transparent';
            node.style.color = textColor;
            node.style.webkitTextFillColor = textColor;
            node.style.border = 'none';
            node.style.boxShadow = 'none';
            node.style.appearance = 'none';
            node.style.webkitAppearance = 'none';
          });

          parentDoc.querySelectorAll('div[data-testid="stNumberInput"] button, div[data-testid="stSelectbox"] svg').forEach((node) => {
            node.style.background = 'transparent';
            node.style.color = textColor;
            node.style.fill = textColor;
            node.style.boxShadow = 'none';
            node.style.border = 'none';
          });
        };

        applyFieldTheme();
        const oldTimer = window.parent.__deepcropcareInputThemeTimer;
        if (oldTimer) clearInterval(oldTimer);
        window.parent.__deepcropcareInputThemeTimer = setInterval(applyFieldTheme, 500);
        </script>
        """,
        height=0,
    )


def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    if model is None:
        return None

    if not last_conv_layer_name:
        last_conv_layer_name = DEFAULT_GRADCAM_LAYER

    try:
        base_model_name, target_layer_name = last_conv_layer_name.split("/", 1)
        base_model = model.get_layer(base_model_name)
        target_layer = base_model.get_layer(target_layer_name)
    except Exception:
        try:
            base_model_name, target_layer_name = DEFAULT_GRADCAM_LAYER.split("/", 1)
            base_model = model.get_layer(base_model_name)
            target_layer = base_model.get_layer(target_layer_name)
        except Exception:
            return None

    if not isinstance(base_model, tf.keras.Model):
        return None

    def _unwrap_tensor(value):
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return value[0]
        return value

    def _prepare_model_inputs(model_inputs, array):
        if isinstance(model_inputs, dict):
            first_key = next(iter(model_inputs.keys()), None)
            return {first_key: array} if first_key else array
        if isinstance(model_inputs, (list, tuple)):
            return [array]
        return array

    base_inputs = getattr(base_model, "inputs", None)
    target_output = _unwrap_tensor(getattr(target_layer, "output", None))
    base_output = _unwrap_tensor(getattr(base_model, "output", None))
    if base_inputs is None or target_output is None or base_output is None:
        return None

    try:
        base_feature_model = tf.keras.models.Model(
            base_inputs,
            [target_output, base_output],
        )
    except Exception:
        return None

    classifier_input = tf.keras.Input(shape=base_output.shape[1:])
    x = classifier_input
    base_index = model.layers.index(base_model)
    for layer in model.layers[base_index + 1:]:
        if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)):
            x = layer(x, training=False)
        else:
            x = layer(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        prepared_inputs = _prepare_model_inputs(base_inputs, img_array)
        conv_outputs, base_outputs = base_feature_model(prepared_inputs, training=False)
        if conv_outputs is None or base_outputs is None:
            return None
        tape.watch(conv_outputs)
        predictions = classifier_model(base_outputs, training=False)
        if predictions is None:
            return None
        predictions = tf.squeeze(predictions)
        if pred_index is None:
            pred_index = tf.argmax(predictions)
        class_channel = predictions[pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None
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
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    if heatmap.ndim != 2:
        raise ValueError(f"Unexpected heatmap shape: {heatmap.shape}")
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)


def preprocess_disease_image(image, img_size=(224, 224)):
    img_resized = image.resize(img_size)
    img_arr = img_to_array(img_resized)
    img_arr = preprocess_input(img_arr)
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_resized, img_arr


def _find_last_conv_layer_in_model(model):
    try:
        model.get_layer("mobilenetv2_1.00_224").get_layer("out_relu")
        return DEFAULT_GRADCAM_LAYER
    except Exception:
        pass

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for nested_layer in reversed(layer.layers):
                if isinstance(nested_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return f"{layer.name}/{nested_layer.name}"
                output_shape = getattr(getattr(nested_layer, "output", None), "shape", None)
                if output_shape is not None and len(output_shape) == 4:
                    if not any(token in nested_layer.name.lower() for token in ["flatten", "gap", "pool", "input"]):
                        return f"{layer.name}/{nested_layer.name}"
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        output_shape = getattr(layer, "output_shape", None)
        if isinstance(output_shape, tuple) and len(output_shape) == 4:
            if not any(token in layer.name.lower() for token in ["flatten", "gap", "pool"]):
                return layer.name
    return None


def load_disease_model():
    model_path = "training_outputs/plant_disease_mobilenetv2.h5"
    if os.path.exists(model_path):
        try:
            return load_model(model_path, compile=False), model_path
        except Exception:
            pass
    return None, None


def get_weather(city_name, lang):
    api_key = "8c3a497f31607fe66be1f23c65538904"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        payload = response.json()
        return payload["main"]["temp"], payload["main"]["humidity"], None
    except Exception:
        return 25.0, 70.0, t("weather_unavailable", lang)


@st.cache_resource
def load_resources(cache_version=GRADCAM_CACHE_VERSION):
    _ = cache_version
    disease_model, disease_model_path = load_disease_model()

    detected_name = None
    if disease_model:
        detected_name = _find_last_conv_layer_in_model(disease_model)
        if not detected_name:
            detected_name = DEFAULT_GRADCAM_LAYER

    try:
        crop_model = joblib.load("rf_crop_recommendation.joblib")
    except Exception:
        crop_model = None

    return disease_model, crop_model, detected_name, disease_model_path


disease_model, crop_model, detected_conv_name, disease_model_path = load_resources()


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 5.6rem !important;
        max-width: 95%;
    }
    .stApp {
        background: linear-gradient(180deg, #1c6cf0 0%, #1763e7 42%, #0f1117 100%);
        background-attachment: fixed;
        color: #eef6ee;
    }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #eef6ee;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("https://www.transparenttextures.com/patterns/leaf.png");
        opacity: 0.03;
        pointer-events: none;
    }
    .top-header {
        position: relative;
        overflow: hidden;
        min-height: 0;
        border-radius: 32px;
        padding: 1.25rem;
        margin-top: 0.35rem;
        background:
            radial-gradient(circle at top left, rgba(93, 245, 173, 0.18), transparent 30%),
            radial-gradient(circle at 85% 15%, rgba(255,255,255,0.14), transparent 20%),
            linear-gradient(135deg, #0d1830 0%, #133e90 46%, #19a06d 100%);
        box-shadow: 0 26px 80px rgba(6, 14, 36, 0.28);
        border: 1px solid rgba(255,255,255,0.12);
        animation: fadeLift 700ms ease both;
    }
    .top-header::before {
        content: "";
        position: absolute;
        inset: -20% auto auto -10%;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(86, 246, 175, 0.18) 0%, rgba(86, 246, 175, 0) 68%);
        filter: blur(10px);
        animation: drift 14s ease-in-out infinite;
        pointer-events: none;
    }
    .top-header::after {
        content: "";
        position: absolute;
        right: -100px;
        bottom: -120px;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.14) 0%, rgba(255,255,255,0) 70%);
        animation: drift 18s ease-in-out infinite reverse;
        pointer-events: none;
    }
    .hero-shell {
        position: relative;
        z-index: 2;
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.9fr);
        gap: 1.2rem;
        align-items: stretch;
    }
    .hero-main,
    .hero-side {
        position: relative;
        overflow: hidden;
        border-radius: 26px;
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .hero-main {
        padding: 1.2rem;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.13), rgba(255,255,255,0.06)),
            linear-gradient(135deg, rgba(10,20,50,0.42), rgba(8,105,88,0.2));
    }
    .hero-side {
        padding: 1.05rem;
        background:
            radial-gradient(circle at top right, rgba(120, 255, 188, 0.18), transparent 28%),
            linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
    }
    .hero-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.3rem;
    }
    .hero-nav-group {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.65rem;
    }
    .hero-nav-item {
        display: inline-flex;
        align-items: center;
        padding: 0.48rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.08);
        color: rgba(245, 248, 240, 0.92) !important;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-decoration: none !important;
        transition: transform 220ms ease, background 220ms ease, border-color 220ms ease;
    }
    .hero-nav-item:hover {
        transform: translateY(-1px);
        background: rgba(255,255,255,0.12);
        border-color: rgba(255,255,255,0.16);
    }
    .hero-brand {
        font-size: clamp(1.45rem, 2.4vw, 2rem);
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.04em;
        color: #f6f7ef !important;
    }
    .hero-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 46px;
        padding: 0.8rem 1.15rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #f8f4e7 0%, #dff7e9 100%);
        color: #103b42 !important;
        font-size: 0.84rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        text-decoration: none !important;
        box-shadow: 0 14px 34px rgba(4, 10, 28, 0.22);
        transition: transform 240ms ease, box-shadow 240ms ease;
    }
    .hero-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 42px rgba(4, 10, 28, 0.28);
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(123, 255, 199, 0.12);
        border: 1px solid rgba(123, 255, 199, 0.18);
        color: #dffbef !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        margin: 0;
        font-size: clamp(2.3rem, 4.2vw, 4rem);
        line-height: 0.96;
        font-weight: 900;
        letter-spacing: -0.06em;
        color: #f8faf5 !important;
        max-width: 12ch;
    }
    .hero-subtext {
        margin: 1rem 0 0;
        max-width: 58ch;
        color: rgba(240, 248, 243, 0.88) !important;
        font-size: clamp(1rem, 1.6vw, 1.15rem);
        line-height: 1.65;
    }
    .hero-actions {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        flex-wrap: wrap;
        margin-top: 1.25rem;
    }
    .hero-cta {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 52px;
        padding: 0.9rem 1.35rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.14);
        color: #f8faf5 !important;
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        text-decoration: none !important;
        transition: transform 220ms ease, background 220ms ease, border-color 220ms ease;
    }
    .hero-cta:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.16);
        border-color: rgba(255,255,255,0.22);
    }
    .hero-secondary {
        color: rgba(233, 245, 238, 0.8) !important;
        font-size: 0.94rem;
        font-weight: 600;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1.1rem;
    }
    .hero-card {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(7, 18, 40, 0.24);
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform 220ms ease, background 220ms ease;
    }
    .hero-card:hover {
        transform: translateY(-2px);
        background: rgba(12, 24, 52, 0.32);
    }
    .hero-card-label {
        display: block;
        margin-bottom: 0.35rem;
        color: rgba(220, 242, 229, 0.7) !important;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-card-value {
        display: block;
        color: #f8faf5 !important;
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.35;
    }
    .hero-side-title {
        margin: 0 0 0.9rem;
        color: #f8faf5 !important;
        font-size: 1.2rem;
        font-weight: 800;
    }
    .hero-side-copy {
        margin: 0;
        color: rgba(235, 245, 240, 0.86) !important;
        line-height: 1.65;
        font-size: 0.98rem;
    }
    .hero-side-stack {
        display: grid;
        gap: 0.85rem;
        margin-top: 1rem;
    }
    .hero-side-panel {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.09);
    }
    .hero-side-panel strong {
        color: #f8faf5 !important;
    }
    @keyframes drift {
        0%, 100% { transform: translate3d(0, 0, 0); }
        50% { transform: translate3d(0, -14px, 0); }
    }
    @keyframes fadeLift {
        0% { opacity: 0; transform: translateY(16px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .language-label {
        color: #d8e6d3;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .hero-minimal {
        position: relative;
        padding: 1.4rem 0 0.4rem;
        animation: fadeLift 680ms ease both;
    }
    .hero-minimal::before {
        content: "";
        position: absolute;
        inset: -12% auto auto -6%;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(129, 255, 184, 0.18) 0%, rgba(129, 255, 184, 0) 70%);
        filter: blur(12px);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(116, 255, 171, 0.1);
        border: 1px solid rgba(116, 255, 171, 0.16);
        color: #def8e6 !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-headline {
        margin: 0;
        font-size: clamp(2.5rem, 4.6vw, 4.5rem);
        line-height: 1;
        letter-spacing: -0.07em;
        color: #f8faf5 !important;
        font-weight: 900;
        max-width: none;
        white-space: nowrap;
        word-break: keep-all;
        overflow-wrap: normal;
    }
    .hero-description {
        margin: 1rem 0 0;
        max-width: 56ch;
        color: rgba(233, 245, 238, 0.86) !important;
        line-height: 1.7;
        font-size: 1.06rem;
    }
    .feature-dock-title {
        margin: 1rem 0 0.65rem;
        color: rgba(230, 245, 235, 0.82) !important;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .feature-stage {
        margin-top: 1rem;
        padding: 1.35rem 1.35rem 1.5rem;
        border-radius: 30px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.1), rgba(255,255,255,0.04)),
            radial-gradient(circle at top right, rgba(118,255,184,0.12), transparent 28%);
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 24px 60px rgba(7, 15, 36, 0.22);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        animation: fadeLift 560ms ease both;
    }
    .feature-stage-title {
        margin: 0;
        color: #f8faf5 !important;
        font-size: clamp(1.8rem, 3vw, 2.6rem);
        font-weight: 900;
        letter-spacing: -0.04em;
    }
    .feature-stage-subtitle {
        margin: 0.35rem 0 1rem;
        color: rgba(233, 245, 238, 0.82) !important;
        max-width: 60ch;
        line-height: 1.65;
    }
    .section-anchor {
        position: relative;
        top: -88px;
    }
    .section-shell {
        margin: 1.4rem 0 1.8rem;
        padding: 1.2rem 1.2rem 1.35rem;
        border-radius: 26px;
        background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.04));
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 18px 48px rgba(6, 14, 36, 0.18);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        animation: fadeLift 520ms ease both;
    }
    .section-kicker {
        display: inline-flex;
        align-items: center;
        margin-bottom: 0.65rem;
        padding: 0.38rem 0.72rem;
        border-radius: 999px;
        background: rgba(120, 255, 188, 0.12);
        border: 1px solid rgba(120, 255, 188, 0.14);
        color: #dffbef !important;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .section-title {
        margin: 0 0 0.25rem;
        color: #f8faf5 !important;
        font-size: clamp(1.75rem, 3vw, 2.7rem);
        font-weight: 900;
        letter-spacing: -0.04em;
    }
    .section-subtitle {
        margin: 0 0 1rem;
        color: rgba(236, 245, 240, 0.82) !important;
        line-height: 1.6;
        max-width: 62ch;
    }
    div.stButton > button,
    div.stDownloadButton > button {
        width: 100% !important;
        white-space: nowrap !important;
        font-weight: bold !important;
        background-color: #28a745 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.55rem 1rem !important;
        font-size: 1rem !important;
        border: none !important;
        transition: transform 220ms ease, background-color 220ms ease, box-shadow 220ms ease !important;
    }
    div.stDownloadButton > button:hover,
    div.stButton > button:hover {
        background-color: #23913c !important;
        color: white !important;
        transform: translateY(-1px);
    }
    div.stDownloadButton > button:focus,
    div.stButton > button:focus {
        box-shadow: 0 0 0 0.18rem rgba(40, 167, 69, 0.28) !important;
        color: white !important;
    }
    .feature-launcher-note {
        margin: 0.15rem 0 0.85rem;
        color: rgba(233, 245, 238, 0.68) !important;
        font-size: 0.92rem;
    }
    .upload-mode-label {
        margin: 0.2rem 0 0.5rem;
        color: rgba(233, 245, 238, 0.78) !important;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .upload-mode-shell {
        margin: 0.25rem 0 0.9rem;
    }
    .upload-mode-shell [role="radiogroup"] {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.65rem;
        width: 100%;
    }
    .upload-mode-shell [role="radiogroup"] > label {
        margin: 0 !important;
        min-height: 84px !important;
        padding: 0.55rem 0.7rem !important;
        border-radius: 22px !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.035)) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        box-shadow: 0 10px 28px rgba(5, 12, 28, 0.12) !important;
        color: rgba(238, 246, 238, 0.88) !important;
        transform: scale(1) translateZ(0);
        transition: transform 110ms ease-out, background 110ms ease-out, border-color 110ms ease-out, box-shadow 110ms ease-out, color 110ms ease-out !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .upload-mode-shell [role="radiogroup"] > label:hover {
        transform: scale(1.015) translateY(-1px);
        background: rgba(255,255,255,0.1) !important;
        border-color: rgba(120,255,188,0.26) !important;
        color: #ffffff !important;
    }
    .upload-mode-shell [role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(180deg, rgba(121, 224, 110, 0.95), rgba(81, 170, 69, 0.96)) !important;
        border: 1px solid rgba(194,255,191,0.92) !important;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.24),
            inset 0 -14px 22px rgba(39, 89, 33, 0.24),
            0 12px 28px rgba(18, 53, 16, 0.24),
            0 0 0 1px rgba(194,255,191,0.12) !important;
        transform: scale(0.965);
        color: #f8fff2 !important;
    }
    .upload-mode-shell [role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    .upload-mode-shell [role="radiogroup"] > label p {
        margin: 0 !important;
        color: inherit !important;
        font-size: 0.96rem !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        text-align: center !important;
    }
    [data-testid="stCameraInput"] > label {
        display: none !important;
    }
    [data-testid="stCameraInput"] {
        margin-top: 0.1rem;
    }
    [data-testid="stCameraInput"] > div {
        background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04)) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 22px !important;
        padding: 0.5rem !important;
        box-shadow: 0 16px 36px rgba(6, 14, 36, 0.16) !important;
    }
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img {
        border-radius: 18px !important;
    }
    [data-testid="stCameraInput"] button {
        background: linear-gradient(180deg, #3cbf67, #23913c) !important;
        color: #f8fff2 !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 20px rgba(18, 53, 16, 0.18) !important;
    }
    [data-baseweb="tab-list"] {
        display: grid !important;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.3rem 0 0.85rem !important;
        padding: 0 !important;
    }
    [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    [data-baseweb="tab-list"] button {
        min-height: 84px !important;
        padding: 0.55rem 0.7rem !important;
        border-radius: 22px !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.035)) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        box-shadow: 0 10px 28px rgba(5, 12, 28, 0.12) !important;
        color: rgba(238, 246, 238, 0.88) !important;
        font-size: 0.9rem !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
        white-space: pre-line !important;
        transform: scale(1) translateZ(0);
        will-change: transform, background, box-shadow, border-color;
        transition: transform 110ms ease-out, background 110ms ease-out, border-color 110ms ease-out, box-shadow 110ms ease-out, color 110ms ease-out !important;
    }
    [data-baseweb="tab-list"] button * {
        color: inherit !important;
    }
    [data-baseweb="tab-list"] button:hover {
        transform: scale(1.015) translateY(-1px);
        background: rgba(255,255,255,0.1) !important;
        border-color: rgba(120,255,188,0.26) !important;
        color: #ffffff !important;
    }
    [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(180deg, rgba(121, 224, 110, 0.95), rgba(81, 170, 69, 0.96)) !important;
        border: 1px solid rgba(194,255,191,0.92) !important;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.24),
            inset 0 -14px 22px rgba(39, 89, 33, 0.24),
            0 12px 28px rgba(18, 53, 16, 0.24),
            0 0 0 1px rgba(194,255,191,0.12) !important;
        transform: scale(0.965);
        color: #f8fff2 !important;
    }
    [data-baseweb="tab-list"] button[aria-selected="true"] * {
        color: #f8fff2 !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stNumberInput"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label {
        color: #eef6ee !important;
    }
    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        transition: transform 220ms ease, border-color 220ms ease, background 220ms ease !important;
    }
    [data-testid="stFileUploader"] section:hover {
        transform: translateY(-1px);
        border-color: rgba(120, 255, 188, 0.28) !important;
        background: rgba(255, 255, 255, 0.09) !important;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {
        opacity: 1 !important;
    }
    [data-testid="stFileUploader"] svg,
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] section div {
        color: #eef6ee !important;
    }
    [data-testid="stFileUploader"] button {
        background: #214d2d !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        color: #eef6ee !important;
    }
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileData"] {
        color: #eef6ee !important;
    }
    [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #eef6ee !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        box-shadow: none !important;
    }
    [data-baseweb="select"] * {
        color: #eef6ee !important;
    }
    [data-testid="stNumberInput"] [data-baseweb="base-input"],
    [data-testid="stTextInput"] [data-baseweb="base-input"],
    [data-testid="stNumberInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    [data-baseweb="input"] > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        box-shadow: none !important;
    }
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        appearance: none !important;
        -webkit-appearance: none !important;
        background-color: transparent !important;
        background-image: none !important;
        border: none !important;
    }
    [data-baseweb="input"] input {
        color: #eef6ee !important;
        background: transparent !important;
        -webkit-text-fill-color: #eef6ee !important;
    }
    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] button:hover,
    [data-testid="stNumberInput"] button:focus,
    [data-testid="stNumberInput"] button:active {
        background: transparent !important;
        color: #eef6ee !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stNumberInput"] button svg,
    [data-testid="stSelectbox"] svg {
        fill: #eef6ee !important;
        color: #eef6ee !important;
    }
    [data-testid="stHeader"] {
        background: rgba(14, 17, 23, 0.78) !important;
    }
    [data-testid="stHeader"] * {
        color: #eef6ee !important;
    }
    button[kind="header"],
    [data-testid="stToolbar"] button,
    [data-testid="stDecoration"] {
        color: #eef6ee !important;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        border-bottom: 4px solid #28a745;
        animation: fadeLift 650ms ease both;
    }
    .prediction-card h2 {
        color: #28a745 !important;
        font-weight: 700 !important;
        margin: 0;
    }
    .prediction-card h3 {
        color: #ffffff !important;
        opacity: 0.92;
        margin: 0.5rem 0 0;
    }
    .result-center {
        max-width: 900px;
        margin: 0 auto;
    }
    .report-summary {
        background: rgba(28, 131, 225, 0.12);
        border: 1px solid rgba(28, 131, 225, 0.35);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin: 1rem 0 1.25rem;
        color: #dcefff;
        box-shadow: 0 10px 30px rgba(10, 54, 95, 0.16);
    }
    .report-summary strong {
        color: #6fc3ff;
    }
    .report-actions {
        display: flex;
        justify-content: center;
        gap: 0.85rem;
        flex-wrap: wrap;
        margin: 0.5rem 0 1rem;
    }
    .report-email-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        padding: 0.7rem 1rem;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 700;
        color: white !important;
        background: #1f7a3e;
    }
    .email-heading {
        margin: 0.85rem 0 0.5rem;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #eef6ee !important;
    }
    .detail-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 15px;
    }
    .detail-card p {
        margin: 0;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #eef6ee !important;
    }
    .stTextInput input {
        color: #eef6ee !important;
    }
    @media (max-width: 900px) {
        .top-header {
            padding: 1rem;
        }
        .hero-shell {
            grid-template-columns: 1fr;
        }
        .hero-nav {
            flex-direction: column;
            align-items: flex-start;
        }
        .hero-title {
            font-size: clamp(2.25rem, 11vw, 3.4rem);
            max-width: none;
        }
        .hero-grid {
            grid-template-columns: 1fr;
        }
        .hero-headline {
            white-space: normal;
        }
        .feature-launcher [data-testid="stButton"] button {
            min-height: 92px !important;
        }
    }
    @media (prefers-color-scheme: light) {
        .stApp {
            background: linear-gradient(180deg, #d8e9ff 0%, #edf4ff 42%, #f7fafc 100%) !important;
            color: #111111 !important;
        }
        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp li,
        .stApp div,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .language-label,
        .email-heading,
        .detail-card p,
        .stTextInput input,
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p,
        [data-testid="stNumberInput"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label,
        [data-baseweb="select"] *,
        [data-baseweb="tab-list"] button,
        [data-testid="stFileUploaderFileName"],
        [data-testid="stFileUploaderFileData"] {
            color: #111111 !important;
            fill: #111111 !important;
        }
        [data-baseweb="tab-list"] button *,
        [data-baseweb="tab-list"] button[aria-selected="true"] * {
            color: inherit !important;
        }
        [data-testid="stFileUploader"] section,
        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] [data-baseweb="base-input"],
        [data-testid="stTextInput"] [data-baseweb="base-input"],
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-baseweb="input"] > div,
        .detail-card {
            background: rgba(12, 18, 14, 0.06) !important;
            border-color: rgba(12, 18, 14, 0.16) !important;
            color: #111111 !important;
        }
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        input,
        textarea {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
            background-color: transparent !important;
        }
        [data-testid="stNumberInput"] button,
        [data-testid="stSelectbox"] svg,
        [data-testid="stFileUploader"] svg,
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] section div {
            color: #111111 !important;
            fill: #111111 !important;
        }
        [data-testid="stFileUploader"] button {
            background: #dceedd !important;
            border: 1px solid rgba(12, 18, 14, 0.16) !important;
        }
        .report-summary {
            color: #111111 !important;
        }
        .report-summary strong {
            color: #0f5f9e !important;
        }
        .top-header {
            background:
                radial-gradient(circle at 50% 0%, rgba(255,255,255,0.4), transparent 24%),
                linear-gradient(135deg, #2c79f4 0%, #1d6bed 45%, #11978d 100%) !important;
        }
        .hero-nav-item,
        .hero-brand,
        .hero-subtext,
        .hero-secondary,
        .hero-card-label,
        .hero-card-value,
        .hero-side-title,
        .hero-side-copy,
        .hero-side-panel,
        .hero-kicker,
        .hero-title {
            color: #f7f4eb !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
inject_input_theme()


if "language" not in st.session_state:
    st.session_state.language = "en"
if "weather_temp" not in st.session_state:
    st.session_state.weather_temp = 25.0
if "weather_hum" not in st.session_state:
    st.session_state.weather_hum = 70.0
if "last_detected_class" not in st.session_state:
    st.session_state.last_detected_class = None
if "last_detection_confidence" not in st.session_state:
    st.session_state.last_detection_confidence = None
if "last_uploaded_signature" not in st.session_state:
    st.session_state.last_uploaded_signature = None
if "disease_result_ready" not in st.session_state:
    st.session_state.disease_result_ready = False
if "disease_upload_mode" not in st.session_state:
    st.session_state.disease_upload_mode = None
if "crop_result" not in st.session_state:
    st.session_state.crop_result = None
if "disease_email_to" not in st.session_state:
    st.session_state.disease_email_to = ""
if "crop_email_to" not in st.session_state:
    st.session_state.crop_email_to = ""
if "last_disease_render_key" not in st.session_state:
    st.session_state.last_disease_render_key = None
if "last_disease_report_text" not in st.session_state:
    st.session_state.last_disease_report_text = ""
if "last_disease_pdf" not in st.session_state:
    st.session_state.last_disease_pdf = None
if "last_disease_report_images" not in st.session_state:
    st.session_state.last_disease_report_images = []
if "last_heatmap_display" not in st.session_state:
    st.session_state.last_heatmap_display = None


def send_email_with_attachment(to_email, subject, body, attachment_bytes, filename):
    smtp_host = st.secrets.get("SMTP_HOST") or os.getenv("SMTP_HOST")
    smtp_port = st.secrets.get("SMTP_PORT") or os.getenv("SMTP_PORT") or 587
    smtp_username = st.secrets.get("SMTP_USERNAME") or os.getenv("SMTP_USERNAME")
    smtp_password = st.secrets.get("SMTP_PASSWORD") or os.getenv("SMTP_PASSWORD")
    smtp_from = st.secrets.get("SMTP_FROM_EMAIL") or os.getenv("SMTP_FROM_EMAIL") or smtp_username
    smtp_use_ssl = str(st.secrets.get("SMTP_USE_SSL") or os.getenv("SMTP_USE_SSL") or "false").lower() == "true"

    if not all([smtp_host, smtp_port, smtp_username, smtp_password, smtp_from]):
        return False, "missing_config"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = smtp_from
    message["To"] = to_email
    message.set_content(body)
    message.add_attachment(attachment_bytes, maintype="application", subtype="pdf", filename=filename)

    try:
        port = int(smtp_port)
        if smtp_use_ssl:
            with smtplib.SMTP_SSL(smtp_host, port, timeout=20) as server:
                server.login(smtp_username, smtp_password)
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_host, port, timeout=20) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(message)
        return True, None
    except Exception as exc:
        return False, str(exc)


def is_email_configured():
    smtp_host = st.secrets.get("SMTP_HOST") or os.getenv("SMTP_HOST")
    smtp_username = st.secrets.get("SMTP_USERNAME") or os.getenv("SMTP_USERNAME")
    smtp_password = st.secrets.get("SMTP_PASSWORD") or os.getenv("SMTP_PASSWORD")
    smtp_from = st.secrets.get("SMTP_FROM_EMAIL") or os.getenv("SMTP_FROM_EMAIL") or smtp_username
    return all([smtp_host, smtp_username, smtp_password, smtp_from])

header_left, header_right = st.columns([5, 1.5], vertical_alignment="top")
with header_right:
    label_placeholder = st.empty()
    selected_language = st.selectbox(
        t("language_selector", st.session_state.language),
        options=list(LANGUAGES.keys()),
        format_func=lambda code: LANGUAGES[code],
        key="language",
        label_visibility="collapsed",
    )
    label_placeholder.markdown(
        f"<div class='language-label'>{t('language_selector', selected_language)}</div>",
        unsafe_allow_html=True,
    )

lang = st.session_state.language

with header_left:
    st.markdown(
        f"""
        <div class="hero-minimal">
            <div class="hero-badge">{t("hero_badge", lang)}</div>
            <h1 class="hero-headline">{t("hero_title", lang)}</h1>
            <p class="hero-description">{t("hero_subtitle", lang)} {t("hero_description", lang)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(f"<div class='feature-dock-title'>{t('choose_workspace', lang)}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='feature-launcher-note'>{t('launcher_note', lang)}</div>", unsafe_allow_html=True)
tab_disease, tab_crop, tab_chat, tab_info = st.tabs(
    [
        f"🌿\n{t('tab_disease', lang)}",
        f"🌾\n{t('tab_crop', lang)}",
        f"💬\n{t('tab_chat', lang)}",
        f"📘\n{t('tab_info', lang)}",
    ]
)

if st.session_state.get("target_tab"):
    inject_tab_switch(st.session_state.target_tab)
    st.session_state.target_tab = None

with tab_disease:
    st.markdown(
        f"""
        <div class='feature-stage'>
            <h2 class='feature-stage-title'>🌿 {t('disease_heading', lang)}</h2>
            <p class='feature-stage-subtitle'>{t('feature_disease_subtitle', lang)}</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='upload-mode-label'>{t('upload_mode_label', lang)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='upload-mode-shell'>", unsafe_allow_html=True)
    previous_upload_mode = st.session_state.disease_upload_mode
    selected_upload_mode = st.radio(
        t("upload_mode_label", lang),
        options=["browse", "camera"],
        index=0 if st.session_state.disease_upload_mode == "browse" else 1 if st.session_state.disease_upload_mode == "camera" else None,
        format_func=lambda value: f"📁 {t('browse_files', lang)}" if value == "browse" else f"📷 {t('camera', lang)}",
        horizontal=True,
        label_visibility="collapsed",
        key="disease_upload_mode",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_upload_mode != previous_upload_mode:
        st.session_state.last_uploaded_signature = None
        st.session_state.disease_result_ready = False
        st.session_state.last_disease_render_key = None
        st.session_state.last_disease_report_text = ""
        st.session_state.last_disease_pdf = None
        st.session_state.last_disease_report_images = []
        st.session_state.last_heatmap_display = None

    uploaded_file = None
    if st.session_state.disease_upload_mode == "camera":
        uploaded_file = st.camera_input(t("capture_leaf", lang), key="disease_camera_capture")
    elif st.session_state.disease_upload_mode == "browse":
        uploaded_file = st.file_uploader(
            t("upload_leaf", lang),
            type=["jpg", "png", "jpeg"],
            key="disease_file_upload",
        )

    if uploaded_file:
        upload_signature = (
            st.session_state.disease_upload_mode,
            uploaded_file.name,
            getattr(uploaded_file, "size", None),
        )
        if upload_signature != st.session_state.last_uploaded_signature:
            st.session_state.last_uploaded_signature = upload_signature
            st.session_state.last_detected_class = None
            st.session_state.last_detection_confidence = None
            st.session_state.disease_result_ready = False
        image = Image.open(uploaded_file).convert("RGB")

        col_s1, col_img, col_s2 = st.columns([1, 0.8, 1])
        with col_img:
            st.image(image, caption=t("uploaded_specimen", lang), width="stretch")

        _, center_col, _ = st.columns([1, 1, 1])
        with center_col:
            run_btn = st.button(t("run_analysis", lang), width="stretch")

        if run_btn:
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.001)
                progress_bar.progress(percent_complete + 1)

            with st.spinner(t("identifying", lang)):
                if disease_model:
                    _, img_arr = preprocess_disease_image(image)

                    prediction = disease_model.predict(img_arr, verbose=0)
                    idx = int(np.argmax(prediction))
                    confidence = float(np.max(prediction) * 100)
                    full_class_name = normalize_predicted_class(CLASS_NAMES[idx])
                    st.session_state.last_detected_disease = disease_display(full_class_name, lang)
                    st.session_state.last_detected_class = full_class_name
                    st.session_state.last_detection_confidence = confidence
                    st.session_state.disease_result_ready = True

                    progress_bar.empty()
                else:
                    progress_bar.empty()
                    st.error(t("disease_model_missing", lang))

        if st.session_state.disease_result_ready and st.session_state.last_detected_class:
            detected_class = st.session_state.last_detected_class
            detected_confidence = st.session_state.last_detection_confidence or 0.0
            detected_class_lower = detected_class.lower()
            heatmap_available = (
                bool(disease_model)
                and "healthy" not in detected_class_lower
                and detected_class != "Background_without_leaves"
            )
            render_key = (
                upload_signature,
                detected_class,
                round(detected_confidence, 4),
                lang,
                bool(heatmap_available),
            )
            if st.session_state.last_disease_render_key != render_key:
                report_images = [(t("original_scan", lang), image.copy())]
                heatmap_bundle = None
                if heatmap_available:
                    try:
                        img_resized, img_arr = preprocess_disease_image(image)
                        heatmap = get_gradcam_heatmap(disease_model, img_arr, detected_conv_name)
                        if heatmap is not None:
                            overlay = overlay_gradcam(img_resized, heatmap)
                            report_images.append((t("infection_hotspots", lang), Image.fromarray(overlay)))
                            heatmap_bundle = {
                                "original": img_resized.copy(),
                                "overlay": overlay.copy(),
                            }
                    except Exception as exc:
                        heatmap_bundle = {"error": str(exc)}

                report_text = build_disease_report_text(
                    detected_class,
                    detected_confidence,
                    lang,
                    uploaded_file.name if uploaded_file else "",
                    heatmap_available,
                )
                st.session_state.last_disease_render_key = render_key
                st.session_state.last_disease_report_text = report_text
                st.session_state.last_disease_report_images = report_images
                st.session_state.last_heatmap_display = heatmap_bundle
                st.session_state.last_disease_pdf = build_pdf_report_bytes(
                    t("report_subject", lang),
                    report_text,
                    lang,
                    report_images,
                )

            report_text = st.session_state.last_disease_report_text
            report_images = st.session_state.last_disease_report_images
            heatmap_bundle = st.session_state.last_heatmap_display
            disease_pdf = st.session_state.last_disease_pdf
            report_filename = f"deepcropcare-report-{detected_class.replace(' ', '-').replace('/', '-')}.txt"
            report_filename = report_filename.replace(".txt", ".pdf")
            disease_email_body = f"{t('report_email_body_intro', lang)}\n\n{report_text}"

            st.markdown("<div class='result-center'>", unsafe_allow_html=True)
            st.markdown(
                f"<br><h3 style='text-align: center;'>{t('analysis_complete', lang)}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class='prediction-card'>
                    <h2>{disease_display(detected_class, lang)}</h2>
                    <h3>{t('confidence', lang)}: {detected_confidence:.2f}%</h3>
                </div>
                <div class='report-summary'>
                    <strong>{t('recommended_action', lang)}:</strong> {disease_advice(detected_class, lang)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            inject_helper_icon(
                ASSISTANT_ICON,
                t("assistant_hint", lang),
                t("assistant_note", lang),
                t("assistant_trigger", lang),
                t("chat_spinner", lang),
                t("tab_disease", lang),
            )
            helper_clicked = st.button(t("assistant_trigger", lang), key="assistant_trigger_button")
            if helper_clicked:
                st.session_state.pending_chat_prompt = build_disease_prompt(detected_class, lang)
                st.session_state.target_tab = t("tab_chat", lang)
                st.rerun()

            if heatmap_available:
                if heatmap_bundle and "overlay" in heatmap_bundle:
                    st.markdown(
                        f"<br><h3 style='text-align: center;'>🎯 {t('heatmap_title', lang)}</h3>",
                        unsafe_allow_html=True,
                    )
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(
                            heatmap_bundle["original"],
                            caption=t("original_scan", lang),
                            width="stretch",
                        )
                    with col_b:
                        st.image(
                            heatmap_bundle["overlay"],
                            caption=t("infection_hotspots", lang),
                            width="stretch",
                        )
                elif heatmap_bundle and heatmap_bundle.get("error"):
                    st.warning(f"Grad-CAM failed while rendering the heatmap: {heatmap_bundle['error']}")
                else:
                    st.warning("Grad-CAM returned no heatmap for this prediction.")
            _, action_col1, _ = st.columns([1.2, 1, 1.2])
            with action_col1:
                st.download_button(
                    t("report_download", lang),
                    data=disease_pdf,
                    file_name=report_filename,
                    mime="application/pdf",
                    width="stretch",
                    key="download_disease_report",
                )
            st.markdown(
                f"<div class='email-heading'>{t('email_report_heading', lang)}</div>",
                unsafe_allow_html=True,
            )
            disease_email_ready = is_email_configured()
            if not disease_email_ready:
                st.info(t("email_config_missing", lang))
            _, disease_email_col, disease_send_col, _ = st.columns([0.85, 1.2, 0.6, 0.85])
            with disease_email_col:
                disease_email = st.text_input(
                    t("email_address", lang),
                    key="disease_email_to",
                    placeholder="name@example.com",
                    label_visibility="collapsed",
                )
            with disease_send_col:
                send_disease_email = st.button(
                    t("send_email", lang),
                    width="stretch",
                    key="send_disease_report_button",
                    disabled=not disease_email_ready,
                )
            if send_disease_email:
                if "@" not in disease_email or "." not in disease_email:
                    st.error(t("email_required", lang))
                else:
                    success, error_code = send_email_with_attachment(
                        disease_email,
                        t("report_subject", lang),
                        disease_email_body,
                        disease_pdf,
                        report_filename,
                    )
                    if success:
                        st.success(t("email_sent", lang))
                    elif error_code == "missing_config":
                        st.info(t("email_config_missing", lang))
                    else:
                        st.error(f"{t('email_failed', lang)}: {error_code}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.session_state.last_uploaded_signature = None
        st.session_state.disease_result_ready = False
        st.session_state.last_disease_render_key = None
        st.session_state.last_disease_report_text = ""
        st.session_state.last_disease_pdf = None
        st.session_state.last_disease_report_images = []
        st.session_state.last_heatmap_display = None
        remove_helper_icon()

    st.markdown("</div>", unsafe_allow_html=True)

with tab_crop:
    remove_helper_icon()
    st.markdown(
        f"""
        <div class='feature-stage'>
            <h2 class='feature-stage-title'>🌾 {t('crop_heading', lang)}</h2>
            <p class='feature-stage-subtitle'>{t('feature_crop_subtitle', lang)}</p>
        """,
        unsafe_allow_html=True,
    )
    col_soil, col_weather = st.columns([1.5, 1])

    with col_soil:
        st.write(f"### 🧪 {t('soil_parameters', lang)}")
        n1, p1, k1 = st.columns(3)
        nitrogen = n1.number_input(t("nitrogen", lang), 0, 200, 50)
        phosphorus = p1.number_input(t("phosphorus", lang), 0, 200, 50)
        potassium = k1.number_input(t("potassium", lang), 0, 200, 50)
        ph = st.slider(t("soil_ph", lang), 0.0, 14.0, 6.5)
        rain = st.number_input(t("rainfall", lang), 0.0, 1000.0, 100.0)

    with col_weather:
        st.write(f"### 🌦️ {t('weather_heading', lang)}")
        city = st.text_input(t("city_input", lang), "Hyderabad")
        if st.button(t("fetch_weather", lang), width="stretch"):
            temp, hum, err = get_weather(city, lang)
            if not err:
                st.session_state.weather_temp = float(temp)
                st.session_state.weather_hum = float(hum)
                st.success(f"{t('weather_success', lang)} {city}")
            else:
                st.error(t("weather_error", lang))

        st.session_state.weather_temp = st.number_input(
            t("temp", lang), value=float(st.session_state.weather_temp), step=0.1
        )
        st.session_state.weather_hum = st.number_input(
            t("humidity", lang), value=float(st.session_state.weather_hum), step=0.1
        )

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        predict_btn = st.button(t("recommend_crop", lang), width="stretch")

    if predict_btn:
        if crop_model:
            features = np.array(
                [[
                    nitrogen,
                    phosphorus,
                    potassium,
                    st.session_state.weather_temp,
                    st.session_state.weather_hum,
                    ph,
                    rain,
                ]]
            )
            prediction_idx = int(crop_model.predict(features)[0])
            crop = LABEL_MAPPING[prediction_idx]
        else:
            st.error(t("crop_model_missing", lang))
            crop = "rice"
        st.session_state.crop_result = {
            "crop": crop,
            "inputs": {
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "ph": ph,
                "rain": rain,
                "temp": st.session_state.weather_temp,
                "hum": st.session_state.weather_hum,
                "city": city,
            },
        }

    if st.session_state.crop_result:
        crop = st.session_state.crop_result["crop"]
        crop_inputs = st.session_state.crop_result["inputs"]
        crop_name = crop_text(crop, "name", lang)
        crop_report_text = build_crop_report_text(crop, lang, crop_inputs)
        crop_pdf = build_pdf_report_bytes(t("crop_report_subject", lang), crop_report_text, lang)
        crop_email_body = f"{t('report_email_body_intro', lang)}\n\n{crop_report_text}"
        st.markdown(
            f"""
            <div class='prediction-card'>
                <h2>🌱 {t('recommended_crop', lang)}: {crop_name.upper()}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        inf1, inf2 = st.columns(2)
        with inf1:
            st.markdown(f"### 📖 {t('description', lang)}")
            st.markdown(
                f"""
                <div class="detail-card" style="border-left: 5px solid #28a745;">
                    <p>
                        {crop_text(crop, 'description', lang)}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="detail-card" style="border-left: 5px solid #1c83e1;">
                    <p><strong>🔍 {t('optimal_conditions', lang)}:</strong> {crop_text(crop, 'conditions', lang)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with inf2:
            st.markdown(f"### 🧪 {t('fertilizer_care', lang)}")
            st.warning(crop_text(crop, "fertilizer", lang))
            st.success(f"**{t('pro_tip', lang)}:** {crop_text(crop, 'tips', lang)}")

        st.markdown(
            f"<br><h3 style='text-align: center;'>✅ {t('crop_analysis_complete', lang)}</h3>",
            unsafe_allow_html=True,
        )
        _, crop_action_col1, _ = st.columns([1.2, 1, 1.2])
        with crop_action_col1:
            st.download_button(
                t("crop_report_download", lang),
                data=crop_pdf,
                file_name=f"deepcropcare-crop-report-{crop}.pdf",
                mime="application/pdf",
                width="stretch",
                key="download_crop_report",
            )
        st.markdown(
            f"<div class='email-heading'>{t('email_report_heading', lang)}</div>",
            unsafe_allow_html=True,
        )
        crop_email_ready = is_email_configured()
        if not crop_email_ready:
            st.info(t("email_config_missing", lang))
        _, crop_email_col, crop_send_col, _ = st.columns([0.85, 1.2, 0.6, 0.85])
        with crop_email_col:
            crop_email = st.text_input(
                t("email_address", lang),
                key="crop_email_to",
                placeholder="name@example.com",
                label_visibility="collapsed",
            )
        with crop_send_col:
            send_crop_email = st.button(
                t("send_email", lang),
                width="stretch",
                key="send_crop_report_button",
                disabled=not crop_email_ready,
            )
        if send_crop_email:
            if "@" not in crop_email or "." not in crop_email:
                st.error(t("email_required", lang))
            else:
                success, error_code = send_email_with_attachment(
                    crop_email,
                    t("crop_report_subject", lang),
                    crop_email_body,
                    crop_pdf,
                    f"deepcropcare-crop-report-{crop}.pdf",
                )
                if success:
                    st.success(t("email_sent", lang))
                elif error_code == "missing_config":
                    st.info(t("email_config_missing", lang))
                else:
                    st.error(f"{t('email_failed', lang)}: {error_code}")

    st.markdown("</div>", unsafe_allow_html=True)

with tab_chat:
    st.markdown(
        f"""
        <div class='feature-stage'>
            <h2 class='feature-stage-title'>💬 {t('chat_heading', lang)}</h2>
            <p class='feature-stage-subtitle'>{t('feature_chat_subtitle', lang)}</p>
        """,
        unsafe_allow_html=True,
    )
    remove_helper_icon()
    model_id = "gemini-2.5-flash-lite"
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.warning(t("api_missing", lang))

    if st.session_state.get("chat_language") != lang:
        st.session_state.pop("chat_session", None)
        st.session_state.pop("messages", None)
        st.session_state.chat_language = lang

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": t("chat_welcome", lang)}]

    if api_key and "chat_session" not in st.session_state:
        disease_context = st.session_state.get("last_detected_disease", t("general_farming", lang))
        system_instruction = t("system_instruction", lang).format(disease=disease_context)
        st.session_state.chat_session = build_chat_session(api_key, model_id, system_instruction)

    pending_prompt = st.session_state.get("pending_chat_prompt")
    if pending_prompt and st.session_state.get("last_auto_prompt") != pending_prompt:
        st.session_state.messages.append({"role": "user", "content": pending_prompt})
        with st.spinner(t("chat_spinner", lang)):
            try:
                if api_key and "chat_session" in st.session_state:
                    ai_response = chat_session_send_message(st.session_state.chat_session, pending_prompt)
                else:
                    ai_response = build_fallback_disease_report(st.session_state.last_detected_class, lang)
            except Exception:
                ai_response = build_fallback_disease_report(st.session_state.last_detected_class, lang)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.session_state.last_auto_prompt = pending_prompt
            st.session_state.pending_chat_prompt = None
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(t("chat_placeholder", lang))
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner(t("chat_spinner", lang)):
            try:
                if api_key and "chat_session" in st.session_state:
                    ai_response = chat_session_send_message(st.session_state.chat_session, prompt)
                else:
                    ai_response = build_fallback_disease_report(
                        st.session_state.get("last_detected_class") or "Background_without_leaves",
                        lang,
                    )
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                st.rerun()
            except Exception as exc:
                if "429" in str(exc):
                    st.error(t("quota_error", lang))
                else:
                    st.error(f"{t('generic_error', lang)}: {exc}")

    st.divider()
    if st.button(f"🗑️ {t('reset_chat', lang)}"):
        st.session_state.pop("chat_session", None)
        st.session_state.pop("messages", None)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with tab_info:
    remove_helper_icon()
    st.markdown(
        f"""
        <div class='feature-stage'>
            <h2 class='feature-stage-title'>📘 {t('about_heading', lang)}</h2>
            <p class='feature-stage-subtitle'>{t('feature_info_subtitle', lang)}</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        ### 🚀 {t('mission_title', lang)}
        {t('mission_body', lang)}
        """
    )

    st.divider()

    col_cv, col_ml = st.columns(2)
    with col_cv:
        st.markdown(f"#### 🧠 {t('cv_title', lang)}")
        st.write(t("cv_body", lang))

    with col_ml:
        st.markdown(f"#### 📈 {t('ml_title', lang)}")
        st.write(t("ml_body", lang))

    with st.expander(f"🧪 {t('npk_title', lang)}"):
        for line in LANGUAGE_LABELS[lang]["npk_body"]:
            st.write(f"- {line}")

    st.divider()
    st.markdown(f"### 🎯 {t('interpretability', lang)}")
    st.info(
        f"**{t('target_layer', lang)}:** `{detected_conv_name or t('na', lang)}`. {t('target_layer_desc', lang)}"
    )
    st.caption(t("footer", lang))
    st.markdown("</div>", unsafe_allow_html=True)
