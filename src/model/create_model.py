"""Create and save the model artifacts used by the prediction API."""

import json
import logging
import pathlib
import pickle
from typing import List, Tuple

import pandas
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.impute import KNNImputer


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
OUTPUT_DIR = "model"


# Columns from the home sales dataset used for model training.
SALES_COLUMN_SELECTION = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    sales_path: str,
    demographics_path: str,
    sales_column_selection: List[str],
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load and prepare model training data.

    Home sales data is merged with zipcode demographic data. Zipcode is used
    for the merge but is not included as a model feature.

    Args:
        sales_path: Path to the home sales CSV.
        demographics_path: Path to the zipcode demographics CSV.
        sales_column_selection: Columns to load from the sales dataset.

    Returns:
        A tuple containing the model feature DataFrame and target Series.
    """

    logger.info("Loading home sales data from %s", sales_path)

    data = pandas.read_csv(
        sales_path,
        usecols=sales_column_selection,
        dtype={"zipcode": str},
    )

    logger.info(
        "Loaded %d home sales records",
        len(data),
    )

    logger.info(
        "Loading zipcode demographics from %s",
        demographics_path,
    )

    demographics = pandas.read_csv(
        demographics_path,
        dtype={"zipcode": str},
    )

    logger.info(
        "Loaded %d demographic records",
        len(demographics),
    )

    # Merge property data with demographic features.
    merged_data = data.merge(
        demographics,
        how="left",
        on="zipcode",
    )

    # Check whether any sales records could not be matched to demographic
    # information. This catches data quality problems before artifacts are
    # generated.
    demographic_columns = [
        column
        for column in demographics.columns
        if column != "zipcode"
    ]

    unmatched_rows = merged_data[
        demographic_columns
    ].isna().all(axis=1).sum()

    if unmatched_rows > 0:
        raise ValueError(
            f"{unmatched_rows} sales records have no matching "
            "zipcode demographic data"
        )

    # Zipcode is only used for demographic lookup and is not itself
    # a model feature.
    merged_data = merged_data.drop(columns="zipcode")

    # Separate target from model features.
    y = merged_data.pop("price")
    x = merged_data

    logger.info(
        "Prepared training dataset with %d rows and %d features",
        len(x),
        len(x.columns),
    )

    return x, y


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------

def main():
    """Load training data, fit model resources, and save artifacts."""

    logger.info("Starting model artifact generation")

    # Load the feature matrix and prediction target.
    x, y = load_data(
        SALES_PATH,
        DEMOGRAPHICS_PATH,
        SALES_COLUMN_SELECTION,
    )

    # Preserve the existing deterministic train/test split.
    x_train, _x_test, y_train, _y_test = (
        model_selection.train_test_split(
            x,
            y,
            random_state=42,
        )
    )

    logger.info(
        "Training split contains %d records",
        len(x_train),
    )

    # Create the output directory before writing artifacts.
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # KNN imputer
    # -----------------------------------------------------------------------

    logger.info("Fitting KNN imputer")

    # Configuration selected by the imputation experiment.
    #
    # The imputer is fitted only on historical training features.
    # The prediction API uses transform() and never fits on request data.
    imputer = KNNImputer(
        n_neighbors=5,
        weights="distance",
    )

    imputer.fit(x_train)

    imputer_path = output_dir / "imputer.pkl"

    with open(imputer_path, "wb") as imputer_file:
        pickle.dump(
            imputer,
            imputer_file,
        )

    logger.info(
        "Saved KNN imputer to %s",
        imputer_path,
    )

    # -----------------------------------------------------------------------
    # Prediction model
    # -----------------------------------------------------------------------

    logger.info("Training home price prediction model")

    # Preserve the existing prediction model architecture.
    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        neighbors.KNeighborsRegressor(),
    )

    model.fit(
        x_train,
        y_train,
    )

    model_path = output_dir / "model.pkl"

    with open(model_path, "wb") as model_file:
        pickle.dump(
            model,
            model_file,
        )

    logger.info(
        "Saved prediction model to %s",
        model_path,
    )

    # -----------------------------------------------------------------------
    # Feature schema
    # -----------------------------------------------------------------------

    model_features = list(x_train.columns)

    features_path = output_dir / "model_features.json"

    with open(
        features_path,
        "w",
    ) as features_file:
        json.dump(
            model_features,
            features_file,
        )

    logger.info(
        "Saved %d model features to %s",
        len(model_features),
        features_path,
    )

    # -----------------------------------------------------------------------
    # Artifact compatibility checks
    # -----------------------------------------------------------------------

    # The fitted imputer and saved feature schema must describe the same
    # number of features.
    if (
        hasattr(imputer, "n_features_in_")
        and imputer.n_features_in_ != len(model_features)
    ):
        raise ValueError(
            "Generated imputer is incompatible with model feature schema: "
            f"{imputer.n_features_in_} != {len(model_features)}"
        )

    logger.info(
        "Model artifact generation completed successfully"
    )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()

    except Exception:
        logger.exception(
            "Model artifact generation failed"
        )
        raise