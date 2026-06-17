import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 35,
    "tenure_months": 12,
    "monthly_charges": 75.5,
    "support_calls": 2,
    "contract_type": "Monthly",
    "payment_method": "Electronic check",
    "internet_service": "Fiber"
}


def test_root_endpoint():
    """La route / doit retourner status ok."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint():
    """La route /health doit retourner status healthy et model_loaded True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_valid_input():
    """Une requête valide doit retourner 200 avec prediction, label et confidence."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "label" in body
    assert "confidence" in body
    assert body["prediction"] in [0, 1, None]


def test_predict_label_coherent_with_prediction():
    """Le label doit correspondre à la prediction."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    if body["prediction"] == 1:
        assert body["label"] == "churn"
    elif body["prediction"] == 0:
        assert body["label"] == "no_churn"


def test_predict_confidence_between_zero_and_one():
    """La confidence doit être entre 0 et 1."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    if body["confidence"] is not None:
        assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_missing_field():
    """Une requête avec un champ manquant doit retourner 422."""
    response = client.post("/predict", json={"tenure_months": 12})
    assert response.status_code == 422


def test_predict_rejects_wrong_type():
    """Une requête avec un mauvais type doit retourner 422."""
    payload = VALID_PAYLOAD.copy()
    payload["age"] = "trente-cinq"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_contract_type():
    """Un contract_type non autorisé doit retourner 422."""
    payload = VALID_PAYLOAD.copy()
    payload["contract_type"] = "Annuel"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_anomaly_detected():
    """Une valeur hors plage doit retourner un warning et prediction=None."""
    payload = VALID_PAYLOAD.copy()
    payload["age"] = 90
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] is None
    assert "warning" in body
    assert "anomalie" in body["warning"]


def test_metrics_endpoint():
    """/metrics doit retourner les compteurs attendus."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "n_requests" in data
    assert "n_errors" in data
    assert "n_invalid_inputs" in data
    assert "drift_alerts" in data
    assert "predictions_distribution" in data