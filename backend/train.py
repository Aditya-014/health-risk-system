import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# -------------------------------
# Load Dataset
# -------------------------------
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)

# Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# -------------------------------
# Scale Features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train Models
# -------------------------------
lr_model = LogisticRegression(max_iter=1000)
gb_model = GradientBoostingClassifier()

lr_model.fit(X_scaled, y)
gb_model.fit(X_scaled, y)

# -------------------------------
# Save Models (SAFE PATH)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.dump(lr_model, os.path.join(BASE_DIR, "lr_model.pkl"))
joblib.dump(gb_model, os.path.join(BASE_DIR, "gb_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

print("✅ Models trained and saved successfully.")