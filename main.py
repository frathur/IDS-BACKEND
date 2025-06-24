from fastapi import FastAPI, HTTPException
from pydantic import RootModel
from typing import Any, Dict
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="AI-Powered IDS Prediction API")

# --- Load Artifacts ---
MODEL_PATH    = "model/rf_ids_model_top20.pkl"
SCALER_PATH   = "model/scaler_top20.pkl"
FEATURES_PATH = "model/feature_list_top20.pkl"

model    = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)  # e.g. ['Flow Duration', ...]

# Label mapping
label_map = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack – Brute Force",
    13: "Web Attack – SQL Injection",
    14: "Web Attack – XSS"
}

# Define a root model for arbitrary feature dict
class FlowData(RootModel[Dict[str, Any]]):
    """RootModel to accept arbitrary JSON with feature names."""

@app.post("/predict")
def predict(flow: FlowData):
    try:
        # 1. Build DataFrame from incoming dict
        df = pd.DataFrame([flow.root])

        # 2. Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # 3. Check for missing features
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # 4. Subset & reorder
        df = df[features]

        # 5. Ensure numeric dtype
        df = df.astype(float)

        # 6. Scale features
        X_scaled = scaler.transform(df)

        # 7. Predict
        idx = model.predict(X_scaled)[0]
        label = label_map.get(idx, "Unknown")

        return {"prediction": label}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
