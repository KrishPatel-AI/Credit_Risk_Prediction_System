import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_xgboost_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)


def predict_default(input_data: dict):

    df = pd.DataFrame(columns=features)

    df.loc[0] = [0] * len(features)

    for key, value in input_data.items():

        if key in df.columns:

            try:
                df[key] = df[key].astype(float)
                df.at[0, key] = float(value)

            except:
                df.at[0, key] = value

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "default_probability": float(probability)
    }