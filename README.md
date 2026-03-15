# DeepCropCare

DeepCropCare is a multilingual Streamlit app for plant disease diagnosis, crop recommendation, and follow-up agronomy support.

Live app:
- [Open DeepCropCare on Streamlit](https://plantdiseasedetection-mxdqvxema8zfzmu4jkauok.streamlit.app)

It combines:
- a TensorFlow MobileNetV2-based image classifier for leaf disease detection
- Grad-CAM heatmaps for model interpretability
- a crop recommendation model based on soil and weather inputs
- an agronomist chat assistant for farmer-friendly guidance
- PDF and email report generation in English, Hindi, and Telugu

## Features

- Plant disease detection from uploaded leaf images
- Grad-CAM heatmap overlays for diseased leaves
- Crop recommendation using soil NPK, pH, rainfall, temperature, and humidity
- Agronomist AI chat for treatment, prevention, and care advice
- Multilingual UI and PDF reports
- Email delivery for disease and crop reports

## Project Structure

```text
.devcontainer/
fonts/
test/
train/
training_outputs/
validate/
.env
.gitignore
Crop_Recommendation.py
Crop_recommendation.csv
README.md
app.py
icon.jpg
plant_disease.py
plant_disease_model_final4.h5
requirements.txt
rf_crop_recommendation.joblib
runtime.txt
```

Important runtime files:
- `training_outputs/plant_disease_mobilenetv2.h5`: disease classification model
- `training_outputs/class_names.json`: model output class order
- `rf_crop_recommendation.joblib`: crop recommendation model
- `fonts/`: Unicode fonts used for Hindi and Telugu PDF generation
- `requirements.txt`: deployment dependencies
- `runtime.txt`: Streamlit Cloud Python version

## Requirements

Recommended deployment/runtime versions:
- Python `3.11`
- `scikit-learn==1.6.1`

Example `requirements.txt`:

```txt
streamlit
tensorflow
opencv-python-headless
numpy
pillow
requests
python-dotenv
joblib
scikit-learn==1.6.1
fpdf2
uharfbuzz
fonttools
google-genai
```

Example `runtime.txt`:

```txt
python-3.11
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Make sure the trained model files are present in `training_outputs/`.
4. Make sure `rf_crop_recommendation.joblib` is present in the project root.
5. Make sure the font files are present in `fonts/`.
6. Add API keys / SMTP settings if you want chat and email features.
7. Run the Streamlit app.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

## Dataset Layout

The disease dataset is organized into split folders in the repo:

```text
train/
validate/
test/
```

Each split contains class subfolders for the plant disease categories used by the classifier.

## Configuration

The app reads configuration from Streamlit secrets or environment variables.

### Gemini / Agronomist AI

- `GEMINI_API_KEY`

If this is missing, the app falls back to built-in static guidance for disease explanation.

### Email

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`
- `SMTP_USE_SSL`

### Weather

The app fetches weather by location text input. Current default location is `Hyderabad`.

## Fonts

Unicode PDF generation depends on these fonts:

- `fonts/NotoSans-Regular.ttf`
- `fonts/NotoSansDevanagari-Regular.ttf`
- `fonts/NotoSerifTelugu-Regular.ttf`
- `fonts/NotoSerifTelugu-Regular-static.ttf`

These are used so Hindi and Telugu reports render correctly in PDFs, including on Streamlit Cloud.

## Disease Model

The disease detector uses:

- model file: `training_outputs/plant_disease_mobilenetv2.h5`
- class order: `training_outputs/class_names.json`
- alternate older model present in repo: `plant_disease_model_final4.h5`

The app uses `class_names.json` as the source of truth for prediction index mapping.

## Crop Recommendation Model

The app expects a serialized crop recommendation model:

- `rf_crop_recommendation.joblib`
- related files in the repo: `Crop_Recommendation.py`, `Crop_recommendation.csv`

If this file is missing, the disease workflow still works, but crop recommendation falls back to demo/error mode.

## Evaluate Model Accuracy

Use `accuracy.py` to test the disease model on a dataset folder with class subfolders.

Example:

```bash
python accuracy.py --data-dir "/full/path/to/test"
```

Optional arguments:
- `--model-path`
- `--class-names-path`
- `--batch-size`
- `--img-size`

## Notes

- Streamlit Cloud usually runs TensorFlow on CPU, so CUDA/GPU warnings can be ignored.
- For best compatibility, keep the training and deployment `scikit-learn` versions aligned.
- `google-genai` is preferred over the deprecated `google-generativeai` package.

## Purpose

This project is designed to help farmers and agricultural users:
- diagnose visible leaf diseases quickly
- understand infected regions through heatmaps
- get practical fertilizer and care advice
- choose suitable crops based on local conditions
- access guidance in multiple Indian languages
