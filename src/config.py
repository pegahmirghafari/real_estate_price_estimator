"""Application configuration."""

import os


# ---------------------------------------------------------------------------
# Prediction resource paths
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "model/model.pkl",
)

IMPUTER_PATH = os.getenv(
    "IMPUTER_PATH",
    "model/imputer.pkl",
)

MODEL_FEATURES_PATH = os.getenv(
    "MODEL_FEATURES_PATH",
    "model/model_features.json",
)

DEMOGRAPHICS_PATH = os.getenv(
    "DEMOGRAPHICS_PATH",
    "data/zipcode_demographics.csv",
)


# ---------------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0",
)