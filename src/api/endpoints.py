"""API endpoints for home price predictions."""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import (
    DEMOGRAPHICS_PATH,
    IMPUTER_PATH,
    MODEL_FEATURES_PATH,
    MODEL_PATH,
)
from utils.loader import (
    load_demographics,
    load_features,
    load_imputer,
    load_model,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API router
# ---------------------------------------------------------------------------
router = APIRouter()


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class HomeFeatures(BaseModel):
    """Input features for a home price prediction.

    Zipcode is required because it is used to look up demographic features.
    All other property features may be missing and will be imputed.
    """

    bedrooms: float | None = None
    bathrooms: float | None = None
    sqft_living: float | None = None
    sqft_lot: float | None = None
    floors: float | None = None
    sqft_above: float | None = None
    sqft_basement: float | None = None
    zipcode: str


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
@router.get("/health")
async def health_check():
    """Return the current API health status."""

    return {
        "status": "healthy",
    }


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------
@router.post("/predict")
async def predict(home_features: HomeFeatures):
    """Predict a home's sale price.

    The zipcode is used to retrieve demographic information.

    Missing property features are filled using the fitted KNN imputer before
    the completed feature row is passed to the prediction model.
    """

    try:
        # -------------------------------------------------------------------
        # Load prediction resources
        # -------------------------------------------------------------------
        # These loader functions use lru_cache, so resources that were loaded
        # during application startup are reused here rather than being read
        # from disk for every prediction request.
        model = load_model(MODEL_PATH)
        imputer = load_imputer(IMPUTER_PATH)
        model_features = load_features(MODEL_FEATURES_PATH)
        demographics = load_demographics(DEMOGRAPHICS_PATH)

        # -------------------------------------------------------------------
        # Convert request to DataFrame
        # -------------------------------------------------------------------
        input_data = pd.DataFrame(
            [home_features.model_dump()]
        )

        # -------------------------------------------------------------------
        # Retrieve zipcode demographics
        # -------------------------------------------------------------------
        demographic_info = demographics[
            demographics["zipcode"] == home_features.zipcode
        ]

        if demographic_info.empty:
            logger.warning(
                "Prediction requested for unsupported zipcode: %s",
                home_features.zipcode,
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported zipcode: "
                    f"{home_features.zipcode}"
                ),
            )

        # -------------------------------------------------------------------
        # Prepare demographic features
        # -------------------------------------------------------------------
        demographic_info = (
            demographic_info
            .drop(columns="zipcode")
            .reset_index(drop=True)
        )

        input_data = input_data.drop(
            columns="zipcode"
        )

        # -------------------------------------------------------------------
        # Combine home and demographic features
        # -------------------------------------------------------------------
        input_data = pd.concat(
            [
                input_data.reset_index(drop=True),
                demographic_info,
            ],
            axis=1,
        )

        # Ensure the features are in exactly the same order used when the
        # model and imputer artifacts were created.
        input_data = input_data[
            model_features
        ]

        # -------------------------------------------------------------------
        # Impute missing values
        # -------------------------------------------------------------------
        imputed_values = imputer.transform(
            input_data
        )

        imputed_data = pd.DataFrame(
            imputed_values,
            columns=model_features,
        )

        # -------------------------------------------------------------------
        # Generate prediction
        # -------------------------------------------------------------------
        prediction = model.predict(
            imputed_data
        )

        predicted_price = float(
            prediction[0]
        )

        logger.info(
            "Prediction completed successfully for zipcode %s",
            home_features.zipcode,
        )

        return {
            "predicted_price": predicted_price,
        }

    except HTTPException:
        # Preserve intentional API errors such as an unsupported zipcode.
        raise

    except Exception:
        # Log the full traceback internally without exposing implementation
        # details to the API client.
        logger.exception(
            "Prediction failed for zipcode %s",
            home_features.zipcode,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate prediction",
        )