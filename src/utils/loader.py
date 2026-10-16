"""Utilities for loading and caching prediction resources."""
import json
import logging
import pickle
from functools import lru_cache
import pandas as pd
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_model(model_path: str):
    """Load and cache the trained prediction model.
    The model is loaded from disk only once per application process.
    Subsequent calls with the same path return the cached model.
    """
    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        logger.info(
            "Prediction model loaded successfully"
        )
        return model
    except Exception:
        logger.exception(
            "Failed to load prediction model from %s",
            model_path,
        )
        raise
# ---------------------------------------------------------------------------
# Imputer loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_imputer(imputer_path: str):
    """Load and cache the fitted KNN imputer.
    The imputer is fitted offline during model artifact generation.
    At prediction time it is only used to transform missing input values.
    """
    try:
        with open(imputer_path, "rb") as imputer_file:
            imputer = pickle.load(imputer_file)
        logger.info(
            "KNN imputer loaded successfully"
        )
        return imputer
    except Exception:
        logger.exception(
            "Failed to load KNN imputer from %s",
            imputer_path,
        )
        raise
# ---------------------------------------------------------------------------
# Model feature loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_features(features_path: str):
    """Load and cache the model's expected feature names.
    The feature list defines both the expected feature set and the order in
    which features must be passed to the imputer and prediction model.
    """
    try:
        with open(features_path, "r") as features_file:
            features = json.load(features_file)
        logger.info(
            "Model features loaded successfully: %d features",
            len(features),
        )
        return features
    except Exception:
        logger.exception(
            "Failed to load model features from %s",
            features_path,
        )
        raise
# ---------------------------------------------------------------------------
# Demographic data loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_demographics(demographics_path: str):
    """Load and cache zipcode demographic data.
    Zipcodes are loaded as strings so their representation remains consistent
    with the zipcode supplied by the prediction request.
    """
    try:
        demographics = pd.read_csv(
            demographics_path,
            dtype={"zipcode": str},
        )
        logger.info(
            "Demographic data loaded successfully: %d rows",
            len(demographics),
        )
        return demographics
    except Exception:
        logger.exception(
            "Failed to load demographic data from %s",
            demographics_path,
        )
        raise