import pytest
"""Unit tests for the FastAPI endpoints.
These tests verify:
1. The health endpoint is available.
2. Predictions work with complete input data.
3. Predictions work when optional home features are missing.
4. Zipcode remains a required field.
"""

def test_health_endpoint(test_client):
    """Health endpoint should return a healthy status."""

    response = test_client.get("/health")
   
    # The API should be available.
    assert response.status_code == 200
    response_data = response.json()
    
    # Verify the expected health response.
    assert "status" in response_data
    assert response_data["status"] == "healthy"

def test_predict_endpoint_valid_input(test_client, sample_home_features):
    """Prediction should succeed when all home features are provided."""
    response = test_client.post("/predict",json=sample_home_features)

    # A valid request should return successfully.
    assert response.status_code == 200
    response_data = response.json()

    # Verify that the API returned a prediction.
    assert "predicted_price" in response_data

    # The endpoint explicitly converts the prediction to a Python float.
    assert isinstance(response_data["predicted_price"], float)

    # Home prices should be positive.
    assert response_data["predicted_price"] > 0

def test_predict_endpoint_with_missing_values(test_client,sample_home_features):

    """Prediction should succeed when optional features are missing.
    The API should use the fitted KNN imputer to fill missing numerical
    property values before sending the feature row to the prediction model.
    """

    # Start with an otherwise valid request.
    # copy() prevents us from modifying the shared pytest fixture.

    payload = sample_home_features.copy()

    # Simulate missing values in two optional property fields.
    # JSON null values become Python None values.

    payload["bathrooms"] = None
    payload["sqft_lot"] = None
    
    response = test_client.post("/predict", json=payload)

    # Missing optional fields should not cause request validation
    # or prediction to fail.
    assert response.status_code == 200
    response_data = response.json()

    # The API should still return a valid prediction after imputation.
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)
    assert response_data["predicted_price"] > 0

def test_predict_endpoint_with_omitted_values(test_client,sample_home_features):
    """Prediction should succeed when optional fields are omitted entirely.
    This is slightly different from explicitly sending null. The Pydantic
    request model should default omitted optional fields to None, after which
    the KNN imputer should handle the missing values.
    """

    payload = sample_home_features.copy()

    # Completely remove fields from the JSON request instead of sending null.
    payload.pop("bathrooms")
    payload.pop("sqft_lot")

    response = test_client.post("/predict",json=payload)

    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)
    assert response_data["predicted_price"] > 0

def test_predict_endpoint_without_zipcode(test_client,sample_home_features):
    """Prediction should fail when zipcode is missing.
    Zipcode is the only required request field because the API needs it to
    look up demographic information before constructing the model features.
    """

    payload = sample_home_features.copy()
    # Remove the required zipcode.
    payload.pop("zipcode")

    response = test_client.post("/predict",json=payload)

    # FastAPI/Pydantic should reject the request before prediction occurs.
    assert response.status_code == 422

def test_predict_endpoint_with_null_zipcode(test_client,sample_home_features):
    """Prediction should fail when zipcode is explicitly null."""

    payload = sample_home_features.copy()

    # Zipcode cannot be null.
    payload["zipcode"] = None
    response = test_client.post("/predict",json=payload)

    # Pydantic should reject None because zipcode is declared as `str`,
    # rather than `str | None`.
    assert response.status_code == 422


def test_prediction_regression(test_client):
    """Known property should produce a stable prediction.

    This protects against unintended changes to preprocessing, feature
    ordering, imputation, demographics, or model inference.
    """

    payload = {
        "bedrooms": 3,
        "bathrooms": 1,
        "sqft_living": 1180,
        "sqft_lot": 5650,
        "floors": 1,
        "sqft_above": 1180,
        "sqft_basement": 0,
        "zipcode": "98178",
    }

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 200

    response_data = response.json()
    predicted_price = response_data["predicted_price"]

    # This is the known-good prediction from the current production model.
    # A meaningful change here should be reviewed rather than silently accepted.
    assert predicted_price == pytest.approx(
        239142,
        rel=1e-3,
    )

def test_predict_endpoint_with_one_missing_feature(

    test_client,

    sample_home_features,

):

    """Prediction should succeed when one numerical feature is missing."""

    payload = sample_home_features.copy()

    payload["bathrooms"] = None

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 200

    response_data = response.json()

    assert "predicted_price" in response_data

    assert isinstance(response_data["predicted_price"], float)

    assert response_data["predicted_price"] > 0
def test_predict_endpoint_rejects_non_numeric_value(

    test_client,

    sample_home_features,

):

    """Non-numeric values for numerical features should fail validation."""

    payload = sample_home_features.copy()

    payload["bedrooms"] = "not-a-number"

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 422
def test_predict_endpoint_rejects_unknown_zipcode(

    test_client,

    sample_home_features,

):

    """A zipcode without demographic data should be rejected."""

    payload = sample_home_features.copy()

    payload["zipcode"] = "00000"

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 422

def test_prediction_response_schema(
    test_client,
    sample_home_features,
):

    """Prediction response should preserve its public API contract."""

    response = test_client.post("/predict", json=sample_home_features)

    assert response.status_code == 200

    response_data = response.json()

    assert set(response_data.keys()) == {"predicted_price"}

    assert isinstance(response_data["predicted_price"], float)

