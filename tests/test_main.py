"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""


# ---------------------------------------------------------------------------
# Function Tests
# ---------------------------------------------------------------------------

# TODO 1: Write a test that calls predict_churn() directly with sample features
#         and asserts the result is 0 or 1
#         Hint: import predict_churn from app.model_utils

# TODO 2 (bonus): Write another function test with edge-case inputs


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------

# TODO 3: Write a test that POSTs to /predict with valid JSON
#         and checks the status code and response body
#         Hint: Litestar POST returns 201, not 200
#         Hint: use `with TestClient(app=app) as client:`

# TODO 4: Write a test for GET /health

# TODO 5: Write a test for GET /

# TODO 6 (bonus): Test that invalid input returns status 400
