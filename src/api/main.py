"""Churn Prediction API."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier

from src.data.preprocessor import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    add_features,
    build_pipeline,
)

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using XGBoost.",
    version="1.0.0",
)

MODEL = None
MODEL_PATH = Path("/app/model.pkl")


def _train_dummy_model():
    """Train a lightweight model on synthetic data for demo purposes."""
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "tenure":          rng.integers(0, 72, n),
        "MonthlyCharges":  rng.uniform(20, 120, n),
        "TotalCharges":    rng.uniform(0, 8000, n),
        "gender":          rng.choice(["Male", "Female"], n),
        "Partner":         rng.choice(["Yes", "No"], n),
        "Dependents":      rng.choice(["Yes", "No"], n),
        "PhoneService":    rng.choice(["Yes", "No"], n),
        "MultipleLines":   rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":  rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":    rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection":rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":     rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":     rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract":        rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling":rng.choice(["Yes", "No"], n),
        "PaymentMethod":   rng.choice(["Electronic check", "Mailed check",
                                        "Bank transfer (automatic)",
                                        "Credit card (automatic)"], n),
        "Churn":           rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })
    df = add_features(df)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    pipeline = build_pipeline(XGBClassifier(n_estimators=50, random_state=42,
                                             eval_metric="logloss"))
    pipeline.fit(X, y)
    return pipeline


@app.on_event("startup")
def load_model():
    global MODEL
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
    else:
        MODEL = _train_dummy_model()


class CustomerFeatures(BaseModel):
    tenure: float = 12
    MonthlyCharges: float = 65.0
    TotalCharges: float = 780.0
    gender: str = "Male"
    Partner: str = "Yes"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_segment: str


@app.get("/")
def root():
    return {"service": "churn-prediction-api", "version": "1.0.0", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([customer.model_dump()])
    df = add_features(df)

    prob = float(MODEL.predict_proba(df)[0][1])
    prediction = prob >= 0.5
    segment = "low" if prob < 0.3 else "medium" if prob < 0.6 else "high"

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        risk_segment=segment,
    )
