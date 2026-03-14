import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np


# ----------------------------------------
# Mock DB and model BEFORE importing app
# This is critical — patch at import time
# ----------------------------------------
@pytest.fixture
def client():
    """
    Create a test client with mocked model and DB.
    We mock:
    - joblib.load → fake model
    - SQLAlchemy engine → fake DB
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])  # always predicts class 0

    with patch("app.app.joblib.load", return_value=mock_model), \
         patch("app.app.engine") as mock_engine, \
         patch("app.app.SessionLocal") as mock_session, \
         patch("app.app.Base.metadata.create_all"):

        # Mock DB session
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        # Mock engine connection for health check
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        from app.app import app
        with TestClient(app) as c:
            yield c

def test_home(client):
    """Root endpoint should return welcome message"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Iris ML API"}


def test_predict_success(client):
    """Predict endpoint should return a prediction"""
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)


def test_predict_invalid_input(client):
    """Predict endpoint should reject invalid input"""
    response = client.post("/predict", json={
        "sepal_length": "not_a_number",  # wrong type
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_missing_field(client):
    """Predict endpoint should reject missing fields"""
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        # missing other fields
    })
    assert response.status_code == 422


def test_health_check(client):
    """Health endpoint should return ok"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_health_model_not_loaded(client):
    """Health check should return 500 if model is not loaded"""
    import app.app as app_module
    original_model = app_module.model

    try:
        app_module.model = None
        response = client.get("/health")
        assert response.status_code == 500
        assert response.json()["detail"] == "Model not loaded"
    finally:
        app_module.model = original_model  # restore after test


def test_health_db_unreachable(client):
    """Health check should return 500 if DB is down"""
    with patch("app.app.SessionLocal") as mock_session:
        mock_db = MagicMock()
        mock_db.execute.side_effect = Exception("DB connection failed")
        mock_session.return_value = mock_db

        response = client.get("/health")
        assert response.status_code == 500
        assert response.json()["detail"] == "Database not reachable"