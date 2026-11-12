from pathlib import Path
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "landguard_random_forest.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"
FEATURES = ["rainfall_24h_mm","rainfall_72h_mm","slope_deg","elevation_m",
            "soil_moisture_pct","distance_to_road_m","ndvi","lithology_code"]
app = FastAPI(title="LANDGUARD AI API", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT/"static"), name="static")

class PredictionInput(BaseModel):
    district: str
    rainfall_24h_mm: float = Field(ge=0)
    rainfall_72h_mm: float = Field(ge=0)
    slope_deg: float = Field(ge=0, le=90)
    elevation_m: float = Field(ge=0, le=9000)
    soil_moisture_pct: float = Field(ge=0, le=100)
    distance_to_road_m: float = Field(ge=0)
    ndvi: float = Field(ge=-1, le=1)
    lithology_code: float

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return FileResponse(ROOT/"static"/"index.html")

@app.get("/api/status")
def status():
    ready = MODEL_PATH.exists() and METRICS_PATH.exists()
    metrics = json.loads(METRICS_PATH.read_text()) if ready else {}
    return {"model_ready": ready, "model_name": "Random Forest Classifier" if ready else None,
            "metrics": metrics}

@app.post("/api/predict")
def predict(data: PredictionInput):
    model = load_model()
    if model is None or not METRICS_PATH.exists():
        raise HTTPException(503, "AI model is not trained yet. Follow README: prepare a verified labelled dataset and run train_model.py.")
    row = pd.DataFrame([{k: getattr(data, k) for k in FEATURES}])
    model_obj = model
    pred = int(model_obj.predict(row)[0])
    probs = model_obj.predict_proba(row)[0]
    classes = list(model_obj.classes_)
    # class 1 means landslide inventory / susceptible sample, by dataset contract
    positive_prob = float(probs[classes.index(1)]) if 1 in classes else float(pred)
    metrics = json.loads(METRICS_PATH.read_text())
    category = "Higher" if positive_prob >= .67 else "Moderate" if positive_prob >= .34 else "Lower"
    return {"district": data.district, "susceptibility_class": category,
            "susceptibility_score": round(positive_prob*100,1),
            "explanation": "The classifier combines rainfall, slope, elevation, soil moisture, road distance, vegetation index and lithology patterns learned from the supplied labelled dataset.",
            "model_name": "Random Forest Classifier",
            "validation_balanced_accuracy": round(metrics.get("balanced_accuracy",0)*100,1)}
