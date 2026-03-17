from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Credit Risk Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "credit_risk_xgboost_model.pkl")
features = joblib.load(BASE_DIR / "models" / "model_features.pkl")

features = [str(f) for f in features]


class LoanApplication(BaseModel):
    loan_amnt: float
    int_rate: float
    annual_inc: float
    dti: float


@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API is running"}


@app.post("/predict")
def predict(data: LoanApplication):

    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    df.columns = df.columns.astype(str)

    aligned_df = pd.DataFrame(columns=features)

    for col in df.columns:
        if col in aligned_df.columns:
            aligned_df.loc[0, col] = df.loc[0, col]

    aligned_df = aligned_df.fillna(0)

    aligned_df = aligned_df.apply(pd.to_numeric)

    aligned_df = aligned_df[features]

    prediction = model.predict(aligned_df, validate_features=False)[0]
    probability = model.predict_proba(aligned_df, validate_features=False)[0][1]
    return {
        "prediction": int(prediction),
        "default_probability": float(probability)
    }