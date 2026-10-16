"""Benchmark cached resource access versus repeated disk loading."""

import json
import statistics
import time
import joblib
import pandas as pd
from src.utils.loader import (
    load_demographics,
    load_features,
    load_imputer,
    load_model,
)

ITERATIONS = 100
MODEL_PATH = "src/model/model.pkl"
IMPUTER_PATH = "src/model/imputer.pkl"
FEATURES_PATH = "src/model/model_features.json"
DEMOGRAPHICS_PATH = "src/data/zipcode_demographics.csv"

def benchmark_uncached():
    """Measure loading all prediction resources from disk."""
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        joblib.load(MODEL_PATH)
        joblib.load(IMPUTER_PATH)
        with open(FEATURES_PATH, encoding="utf-8") as file:
            json.load(file)
        pd.read_csv(
            DEMOGRAPHICS_PATH,
            dtype={"zipcode": str},
        )
        times.append(
            (time.perf_counter() - start) * 1000
        )
    return times

def benchmark_cached():
    """Measure access through the application's cached loaders."""
    # Populate the caches before timing.
    load_model(MODEL_PATH)
    load_imputer(IMPUTER_PATH)
    load_features(FEATURES_PATH)
    load_demographics(DEMOGRAPHICS_PATH)
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        load_model(MODEL_PATH)
        load_imputer(IMPUTER_PATH)
        load_features(FEATURES_PATH)
        load_demographics(DEMOGRAPHICS_PATH)
        times.append(
            (time.perf_counter() - start) * 1000
        )
    return times

def summarize(name, times):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Mean:   {statistics.mean(times):.3f} ms")
    print(f"Median: {statistics.median(times):.3f} ms")
    print(f"Min:    {min(times):.3f} ms")
    print(f"Max:    {max(times):.3f} ms")

def main():
    print(f"Running {ITERATIONS} iterations...")
    uncached = benchmark_uncached()
    cached = benchmark_cached()
    summarize("Repeated disk loading", uncached)
    summarize("Cached resource access", cached)
    uncached_mean = statistics.mean(uncached)
    cached_mean = statistics.mean(cached)
    improvement = (
        (uncached_mean - cached_mean)
        / uncached_mean
        * 100
    )
    print("\nComparison")
    print("----------")
    print(
        f"Mean resource-access reduction: "
        f"{improvement:.2f}%"
    )

if __name__ == "__main__":
    main()