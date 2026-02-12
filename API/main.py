"""
API FastAPI para predicción de especies de pingüinos.
Endpoints POST: /rf (Random Forest) y /lr (Logistic Regression).
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Rutas a los modelos (relativas al directorio API)
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")
RF_PATH = os.path.join(MODEL_DIR, "RF.pkl")
LR_PATH = os.path.join(MODEL_DIR, "LR.pkl")

app = FastAPI(
    title="Penguins Species API",
    description="Predicción de especie de pingüino con modelos RF y LR.",
    version="1.0.0",
)

# Carga de modelos al arranque
model_rf = None
model_lr = None


@app.on_event("startup")
def load_models():
    global model_rf, model_lr
    if not os.path.isfile(RF_PATH):
        raise FileNotFoundError(f"No se encontró el modelo RF: {RF_PATH}")
    if not os.path.isfile(LR_PATH):
        raise FileNotFoundError(f"No se encontró el modelo LR: {LR_PATH}")
    model_rf = joblib.load(RF_PATH)
    model_lr = joblib.load(LR_PATH)


class PenguinFeatures(BaseModel):
    """Features requeridas para predecir species (mismo orden que en entrenamiento)."""

    island: str = Field(..., description="Isla: Torgersen, Biscoe o Dream")
    bill_length_mm: float = Field(..., description="Longitud del pico en mm")
    bill_depth_mm: float = Field(..., description="Profundidad del pico en mm")
    flipper_length_mm: float = Field(..., description="Longitud de aleta en mm")
    body_mass_g: float = Field(..., description="Masa corporal en gramos")
    sex: str = Field(..., description="Sexo: male o female")
    year: int = Field(..., description="Año (ej: 2007)")


def predict_with_model(model, features: PenguinFeatures) -> str:
    """Convierte las features a DataFrame y devuelve la especie predicha."""
    row = pd.DataFrame([{
        "island": features.island,
        "bill_length_mm": features.bill_length_mm,
        "bill_depth_mm": features.bill_depth_mm,
        "flipper_length_mm": features.flipper_length_mm,
        "body_mass_g": features.body_mass_g,
        "sex": features.sex,
        "year": features.year,
    }])
    pred = model.predict(row)
    return str(pred[0])


@app.post("/rf")
def predict_rf(features: PenguinFeatures):
    """Predicción de especie usando el modelo Random Forest (RF.pkl)."""
    if model_rf is None:
        raise HTTPException(status_code=503, detail="Modelo RF no cargado")
    try:
        species = predict_with_model(model_rf, features)
        return {"model": "RF", "species": species}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/lr")
def predict_lr(features: PenguinFeatures):
    """Predicción de especie usando el modelo Logistic Regression (LR.pkl)."""
    if model_lr is None:
        raise HTTPException(status_code=503, detail="Modelo LR no cargado")
    try:
        species = predict_with_model(model_lr, features)
        return {"model": "LR", "species": species}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/")
def root():
    """Información de la API."""
    return {
        "message": "Penguins Species API",
        "docs": "/docs",
        "endpoints": {"POST /rf": "Random Forest", "POST /lr": "Logistic Regression"},
    }
