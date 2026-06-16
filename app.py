import json
import logging
import os
import pickle
from collections import deque
from datetime import datetime
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

# ============================================================
# CONFIGURATION LOGGING
# ============================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("churn_api")

# ============================================================
# CHARGEMENT DU MODÈLE DEPUIS LE MODEL REGISTRY
# ============================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
logger.info("Chargement du modèle depuis le MLflow Model Registry...")
model = mlflow.sklearn.load_model("models:/churn-classifier@latest-cindy")
logger.info(f"Modèle chargé : {type(model).__name__}")

# Chargement des colonnes
with open("artifacts/feature_columns.json", "r", encoding="utf-8") as f:
    feature_columns = json.load(f)
logger.info(f"Colonnes chargées : {len(feature_columns)} features")

# Référence pour Evidently (créée à l'étape 11)
reference_sample = None
if os.path.exists("artifacts/reference_sample.csv"):
    reference_sample = pd.read_csv("artifacts/reference_sample.csv")
    logger.info("Référence Evidently chargée.")

# ============================================================
# COMPTEURS DE MÉTRIQUES
# ============================================================
metrics_state = {
    "n_requests": 0,
    "n_errors": 0,
    "n_invalid_inputs": 0,
    "drift_alerts": 0,
    "predictions_distribution": {"churn": 0, "no_churn": 0},
    "evidently_reports_generated": 0,
}

# ============================================================
# DÉTECTION DE DÉRIVE MAISON (Étape 10)
# ============================================================
TRAIN_STATS = {
    "tenure_months": {"mean": 32.4, "std": 24.5},
    "monthly_charges": {"mean": 64.8, "std": 30.1},
}

WINDOW_SIZE = 50
recent_inputs = {var: deque(maxlen=WINDOW_SIZE) for var in TRAIN_STATS}

# Buffer pour Evidently — réduit car dataset petit
drift_buffer = deque(maxlen=20)

# ============================================================
# PLAGES D'ENTRAÎNEMENT POUR DÉTECTION D'ANOMALIES (Étape 9)
# ============================================================
TRAIN_RANGES = {
    "tenure_months": (0, 72),
    "monthly_charges": (18.0, 120.0),
    "age": (18, 80),
    "support_calls": (0, 10),
}

# ============================================================
# SCHÉMA D'ENTRÉE PYDANTIC
# ============================================================
class CustomerInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Âge du client")
    tenure_months: int = Field(..., ge=0, le=120, description="Ancienneté en mois")
    monthly_charges: float = Field(..., ge=0, description="Charges mensuelles")
    support_calls: int = Field(..., ge=0, le=50, description="Nombre d'appels support")
    contract_type: Literal["Monthly", "One year", "Two year"] = Field(
        ..., description="Type de contrat"
    )
    payment_method: Literal[
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ] = Field(..., description="Mode de paiement")
    internet_service: Literal["DSL", "Fiber", "None"] = Field(
        ..., description="Type de service internet"
    )

    @model_validator(mode="after")
    def check_coherence(self):
        if self.tenure_months > 0 and self.monthly_charges == 0:
            raise ValueError(
                "monthly_charges=0 incohérent avec tenure_months > 0"
            )
        return self


# ============================================================
# FONCTIONS DE DÉTECTION
# ============================================================
def detect_anomaly(payload: CustomerInput) -> list:
    """Détecte les valeurs hors plage d'entraînement."""
    anomalies = []
    for var, (low, high) in TRAIN_RANGES.items():
        value = getattr(payload, var)
        if value < low or value > high:
            anomalies.append(f"{var}={value} hors plage [{low}, {high}]")
    return anomalies


def check_drift(payload: CustomerInput) -> dict:
    """Détecte une dérive sur fenêtre glissante."""
    drift_report = {}
    for var, stats in TRAIN_STATS.items():
        recent_inputs[var].append(getattr(payload, var))
        if len(recent_inputs[var]) >= WINDOW_SIZE:
            recent_mean = sum(recent_inputs[var]) / len(recent_inputs[var])
            gap = abs(recent_mean - stats["mean"]) / stats["std"]
            if gap > 0.5:
                drift_report[var] = round(gap, 2)
    return drift_report


# ============================================================
# APPLICATION FASTAPI
# ============================================================
app = FastAPI(
    title="Churn Prediction API",
    description="API d'inférence pour la prédiction du churn client",
    version="1.0"
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Churn Prediction API"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features": len(feature_columns)
    }


@app.get("/metrics")
def metrics():
    return metrics_state


@app.post("/predict")
def predict(payload: CustomerInput):
    metrics_state["n_requests"] += 1
    logger.info(
        f"Requête reçue : tenure_months={payload.tenure_months}, "
        f"monthly_charges={payload.monthly_charges}, "
        f"contract_type={payload.contract_type}"
    )

    # Détection d'anomalie métier
    anomalies = detect_anomaly(payload)
    if anomalies:
        logger.warning(f"Entrée anormale détectée : {anomalies}")
        metrics_state["n_invalid_inputs"] += 1
        return {
            "prediction": None,
            "label": None,
            "confidence": None,
            "warning": "anomalie détectée",
            "details": anomalies,
        }

    try:
        # Préparation des données
        df = pd.DataFrame([payload.model_dump()])
        df_encoded = pd.get_dummies(df, drop_first=True)
        df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # Ajout au buffer Evidently
        drift_buffer.append(df_aligned.iloc[0].to_dict())

        # Prédiction
        prediction = model.predict(df_aligned)[0]
        proba = model.predict_proba(df_aligned)[0].max()
        label = "churn" if prediction == 1 else "no_churn"

        # Mise à jour des compteurs
        metrics_state["predictions_distribution"][label] += 1

        # Détection de dérive maison
        drift = check_drift(payload)
        if drift:
            logger.warning(
                f"Dérive détectée sur {WINDOW_SIZE} requêtes : {drift}"
            )
            metrics_state["drift_alerts"] += 1

        logger.info(
            f"Prédiction : {label} (confiance={proba:.2f})"
        )

        return {
            "prediction": int(prediction),
            "label": label,
            "confidence": round(float(proba), 4),
            "drift_alert": drift if drift else None,
        }

    except Exception as e:
        metrics_state["n_errors"] += 1
        logger.error(f"Échec de prédiction : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drift_report")
def drift_report():
    """Génère un rapport Evidently sur les 100 dernières requêtes."""
    if reference_sample is None:
        raise HTTPException(
            status_code=503,
            detail="Référence Evidently non disponible (artifacts/reference_sample.csv manquant)."
        )

    if len(drift_buffer) < 20:
        raise HTTPException(
            status_code=425,
            detail=f"Buffer : {len(drift_buffer)}/20. Envoie plus de requêtes /predict."
        )

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        current = pd.DataFrame(list(drift_buffer))
        result = Report(metrics=[DataDriftPreset()]).run(
            reference_data=reference_sample,
            current_data=current
        )

        os.makedirs("logs/drift_reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"logs/drift_reports/drift_{timestamp}.html"
        result.save_html(path)

        metrics_state["evidently_reports_generated"] += 1
        logger.info(f"Rapport Evidently généré : {path}")

        return {
            "status": "report_generated",
            "report_path": path
        }

    except Exception as e:
        logger.error(f"Erreur génération rapport Evidently : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))