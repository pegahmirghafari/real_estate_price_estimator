import pandas as pd
import pytest


API_INPUT_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]


@pytest.mark.integration
def test_predict_endpoint_integration(http_client, sample_home_features):
    """Test the /predict endpoint via HTTP using the Dockerized API."""
    response = http_client.post("/predict", json=sample_home_features)

    assert response.status_code == 200

    response_data = response.json()
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)
    assert response_data["predicted_price"] > 0


@pytest.mark.integration
def test_health_endpoint_integration(http_client):
    """Test the /health endpoint via HTTP using the Dockerized API."""
    response = http_client.get("/health")

    assert response.status_code == 200

    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"


@pytest.mark.integration
def test_predict_endpoint_with_missing_values_integration(
    http_client,
    sample_home_features,
):
    """Dockerized API should predict successfully with missing features."""
    payload = sample_home_features.copy()

    # These values should be filled by the fitted KNN imputer.
    payload["bathrooms"] = None
    payload["sqft_lot"] = None

    response = http_client.post("/predict", json=payload)

    assert response.status_code == 200

    response_data = response.json()
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)
    assert response_data["predicted_price"] > 0


@pytest.mark.integration
def test_future_unseen_examples_integration(http_client):
    """All supplied unseen examples should produce valid predictions."""
    examples = pd.read_csv(
        "/app/src/data/future_unseen_examples.csv"
    )

    assert len(examples) > 0

    for _, row in examples.iterrows():
        payload = row[API_INPUT_COLUMNS].to_dict()

        # The API request schema expects zipcode as a string.
        payload["zipcode"] = str(int(payload["zipcode"]))

        response = http_client.post(
            "/predict",
            json=payload,
        )

        assert response.status_code == 200

        response_data = response.json()

        assert set(response_data.keys()) == {"predicted_price"}
        assert isinstance(response_data["predicted_price"], float)
        assert response_data["predicted_price"] > 0


@pytest.mark.integration
def test_future_unseen_examples_with_missing_values_integration(
    http_client,
):
    """Unseen examples should predict successfully after introducing missing data."""
    examples = pd.read_csv(
        "/app/src/data/future_unseen_examples.csv"
    ).head(10)

    assert len(examples) > 0

    for index, row in examples.iterrows():
        payload = row[API_INPUT_COLUMNS].to_dict()

        payload["zipcode"] = str(int(payload["zipcode"]))

        # Deliberately introduce different missing-data patterns.
        if index % 2 == 0:
            payload["bathrooms"] = None
        else:
            payload["sqft_lot"] = None
            payload["sqft_basement"] = None

        response = http_client.post(
            "/predict",
            json=payload,
        )

        assert response.status_code == 200

        response_data = response.json()

        assert set(response_data.keys()) == {"predicted_price"}
        assert isinstance(response_data["predicted_price"], float)
        assert response_data["predicted_price"] > 0