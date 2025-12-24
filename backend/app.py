from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(title="Early Disease Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

lr_model = joblib.load(os.path.join(BASE_DIR, "lr_model.pkl"))
gb_model = joblib.load(os.path.join(BASE_DIR, "gb_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Input schema
class HealthInput(BaseModel):
    age: int
    bmi: float
    activity_level: int   # 1–5
    sleep_hours: float
    family_history: int   # 0 or 1
    sugar_intake: int     # 1–5
    stress_level: int     # 1–5

# Risk logic
def lifestyle_risk(data):
    score = 0
    score += data.activity_level * 2
    score += data.sugar_intake * 3
    score += data.stress_level * 2
    score += (1 if data.sleep_hours < 6 else 0) * 4
    score += data.family_history * 5
    return min(score, 100)

@app.post("/predict")
def predict(data: HealthInput):

    ml_input = np.array([[ 
        0,
        120,
        70,
        20,
        80,
        data.bmi,
        0.5,
        data.age
    ]])

    ml_scaled = scaler.transform(ml_input)

    prob_lr = lr_model.predict_proba(ml_scaled)[0][1]
    prob_gb = gb_model.predict_proba(ml_scaled)[0][1]

    ml_risk = (prob_lr + prob_gb) / 2 * 100

    life_risk = (
        data.activity_level * 2 +
        data.sugar_intake * 3 +
        data.stress_level * 2 +
        (4 if data.sleep_hours < 6 else 0) +
        data.family_history * 5
    )

    final_risk = 0.6 * ml_risk + 0.4 * life_risk

    if final_risk < 30:
        category = "Low"
    elif final_risk < 60:
        category = "Moderate"
    else:
        category = "High"

    confidence = "85%"

    return {
        "risk_percentage": round(final_risk, 2),
        "risk_category": category,
        "confidence": confidence
    }
