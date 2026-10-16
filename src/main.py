"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints import router as api_router
from config import (
    APP_VERSION,
    DEMOGRAPHICS_PATH,
    IMPUTER_PATH,
    LOG_LEVEL,
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
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load prediction resources when the application starts.

    Preloading ensures that missing or corrupted model artifacts are detected
    during application startup rather than during the first prediction
    request.

    The loader functions use lru_cache, so resources loaded here are reused
    by the prediction endpoint without being read from disk again.
    """

    logger.info("Starting home price prediction API")

    try:
        logger.info("Loading prediction resources")

        # Preload the prediction model.
        #
        # The returned model is not used directly here. Calling load_model()
        # during startup ensures that the model exists, can be deserialized,
        # and is placed in the loader cache before requests are accepted.
        load_model(MODEL_PATH)

        # Preload the remaining prediction resources.
        imputer = load_imputer(IMPUTER_PATH)
        model_features = load_features(MODEL_FEATURES_PATH)
        demographics = load_demographics(DEMOGRAPHICS_PATH)

        # Verify that the fitted imputer expects the same number of features
        # defined by the model feature schema.
        if (
            hasattr(imputer, "n_features_in_")
            and imputer.n_features_in_ != len(model_features)
        ):
            raise ValueError(
                "Imputer feature count does not match model feature count: "
                f"{imputer.n_features_in_} != {len(model_features)}"
            )

        logger.info(
            "Prediction resources loaded successfully: "
            "%d model features, %d demographic records",
            len(model_features),
            len(demographics),
        )

    except Exception:
        # The application should fail during startup if a required prediction
        # resource is missing, corrupt, or incompatible. This prevents the
        # service from accepting requests when it cannot safely predict.
        logger.exception(
            "Application startup failed while loading prediction resources"
        )
        raise

    # Application is ready to receive requests.
    yield

    logger.info("Shutting down home price prediction API")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Seattle Home Value Prediction API",
    description=(
        "REST API for predicting Seattle-area home values. "
        "Missing property features are handled using KNN imputation."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
app.include_router(api_router)


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )