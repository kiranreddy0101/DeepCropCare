import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --------------------------
# 1. Load Dataset
# --------------------------
# CSV should have columns: N, P, K, temperature, humidity, ph, rainfall, label
df = pd.read_csv("/content/drive/MyDrive/Crop_recommendation.csv")  # Change to your file path

# --------------------------
# 2. Label Mapping
# --------------------------
label_mapping = {
    0: "rice", 1: "wheat", 2: "maize", 3: "chickpea", 4: "kidneybeans",
    5: "pigeonpeas", 6: "mothbeans", 7: "mungbean", 8: "blackgram", 9: "lentil",
    10: "pomegranate", 11: "banana", 12: "mango", 13: "grapes", 14: "watermelon",
    15: "muskmelon", 16: "apple", 17: "orange", 18: "papaya", 19: "coconut",
    20: "cotton", 21: "jute", 22: "coffee"
}
reverse_mapping = {v: k for k, v in label_mapping.items()}

# Ensure labels are numeric
if df['label'].dtype == object:
    df['label'] = df['label'].map(reverse_mapping)

# --------------------------
# 3. Features & Labels
# --------------------------
feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[feature_order].to_numpy()  # exact order as during training
y = df['label'].to_numpy()

# --------------------------
# 4. Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------
# 5. Train Model
# --------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --------------------------
# 6. Test Accuracy
# --------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Classification Report with crop names
y_test_names = [label_mapping[i] for i in y_test]
y_pred_names = [label_mapping[i] for i in y_pred]
print("\n📊 Classification Report:")
print(classification_report(y_test_names, y_pred_names))

# --------------------------
# 7. Save Model
# --------------------------
joblib.dump(model, "/content/drive/MyDrive/rf_crop_recommendation.joblib")
print("\n💾 Model saved as rf_crop_recommendation.joblib")

# --------------------------
# 8. Save Mismatches
# --------------------------
test_df = pd.DataFrame(X_test, columns=feature_order)
test_df['actual'] = y_test_names
test_df['predicted'] = y_pred_names
mismatches = test_df[test_df['actual'] != test_df['predicted']]
mismatches.to_csv("/content/mismatches.csv", index=False)
print(f"⚠️ Mismatches saved to /content/mismatches.csv")
