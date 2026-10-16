"""Tests for compatibility between production model artifacts."""

from utils.loader import load_features, load_imputer, load_model

def test_imputer_matches_model_features():
    """Imputer should use the same number of features as the model schema."""
    imputer = load_imputer("model/imputer.pkl")
    model_features = load_features("model/model_features.json")
    assert imputer.n_features_in_ == len(model_features)

def test_model_feature_order_is_correct():
    """Production model features should remain in the expected order."""
    model_features = load_features("model/model_features.json")
    expected_features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
        "ppltn_qty",
        "urbn_ppltn_qty",
        "sbrbn_ppltn_qty",
        "farm_ppltn_qty",
        "non_farm_qty",
        "medn_hshld_incm_amt",
        "medn_incm_per_prsn_amt",
        "hous_val_amt",
        "edctn_less_than_9_qty",
        "edctn_9_12_qty",
        "edctn_high_schl_qty",
        "edctn_some_clg_qty",
        "edctn_assoc_dgre_qty",
        "edctn_bchlr_dgre_qty",
        "edctn_prfsnl_qty",
        "per_urbn",
        "per_sbrbn",
        "per_farm",
        "per_non_farm",
        "per_less_than_9",
        "per_9_to_12",
        "per_hsd",
        "per_some_clg",
        "per_assoc",
        "per_bchlr",
        "per_prfsnl",
    ]
    assert model_features == expected_features

def test_model_loader_uses_cache():
    """Repeated model loads should return the cached model instance."""
    load_model.cache_clear()
    first_model = load_model("model/model.pkl")
    cache_after_first_load = load_model.cache_info()
    second_model = load_model("model/model.pkl")
    cache_after_second_load = load_model.cache_info()
    # The same in-memory model object should be returned.
    assert first_model is second_model
    # First call should miss the cache; second call should hit it.
    assert cache_after_first_load.misses == 1
    assert cache_after_second_load.hits == 1


    