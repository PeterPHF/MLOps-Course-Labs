"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""
from app.model_utils import predict_churn 
from main import app
from litestar.testing import TestClient
# ---------------------------------------------------------------------------
# Function Tests
# ---------------------------------------------------------------------------

# TODO 1: Write a test that calls predict_churn() directly with sample features
#         and asserts the result is 0 or 1
#         Hint: import predict_churn from app.model_utils
def test_predict_churn_returns_binary_label() -> None:
    sample_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.2]
    prediction = predict_churn(sample_features)

    assert prediction in (0, 1)


# TODO 2 (bonus): Write another function test with a `with pytest.raises(...):`


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------

# TODO 3: Write a test that POSTs to /predict with valid JSON
#         and checks the status code and response body
#         Hint: Litestar POST returns 201, not 200
#         Hint: use `with TestClient(app=app) as client:`

def test_predict_endpoint_returns_prediction() -> None:
    payload = {
        "Surname": 0.1,
        "CreditScore": 0.2,
        "Geography": 0.3,
        "Gender": 0.4,
        "Age": 0.5,
        "Tenure": 0.6,
        "Balance": 0.7,
        "NumOfProducts": 0.8,
        "HasCrCard": 0.9,
        "IsActiveMember": 1.0,
        "EstimatedSalary": 0.2,
    }

    with TestClient(app=app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 201
    assert response.json()["prediction"] in (0, 1)

# TODO 4: Write a test for GET /health
def test_health_endpoint() -> None:
    with TestClient(app=app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# TODO 5: Write a test for GET /
def test_home_endpoint() -> None:
    with TestClient(app=app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Churn Prediction API"}

# TODO 6 (bonus): Test that invalid input returns status 400
def test_predict_endpoint_invalid_input() -> None:
    payload = {
        "Surname": "",
        "CreditScore": 0.2,
        "Geography": 0.3,
        "Gender": 0.4,
        "Age": 0.5,
        "Tenure": 0.6,
        "Balance": 0.7,
        "NumOfProducts": 0.8,
        "HasCrCard": 0.9,
        "IsActiveMember": 1.0,
        "EstimatedSalary": 0.2,
    }
    
    with TestClient(app=app) as client:
        response = client.post("/predict", json=payload)
        
    assert response.status_code == 400
        
