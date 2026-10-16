"""Simple benchmark for the home price prediction API."""

import os
import statistics
import time
import httpx2

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_COUNT = 100

PAYLOAD = {
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1500.0,
    "sqft_lot": 5000.0,
    "floors": 1.0,
    "sqft_above": 1200.0,
    "sqft_basement": 300.0,
    "zipcode": "98042",
}

def percentile(values, percentile_value):
    """Return a percentile without requiring NumPy."""
    values = sorted(values)
    index = int((percentile_value / 100) * (len(values) - 1))
    return values[index]

def main():
    latencies = []
    with httpx2.Client(
        base_url=API_BASE_URL,
        timeout=10.0,
    ) as client:
        print(f"Benchmarking: {API_BASE_URL}")
        print("Warming up API...")

        # Warm-up requests are deliberately excluded from measurements.
        for _ in range(5):
            response = client.post("/predict", json=PAYLOAD)
            response.raise_for_status()
        print(f"Sending {REQUEST_COUNT} measured requests...")       
        successful_requests = 0

        for _ in range(REQUEST_COUNT):
            start = time.perf_counter()
            response = client.post(
                "/predict",
                json=PAYLOAD,
            )
            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            if response.status_code == 200:
                successful_requests += 1
                latencies.append(elapsed_ms)

        if not latencies:
            raise RuntimeError(
                "Benchmark completed without any successful requests."
            )
        
    print("\nBenchmark results")
    print("-----------------")
    print(f"Requests:      {REQUEST_COUNT}")
    print(f"Successful:    {successful_requests}")
    print(f"Mean latency:  {statistics.mean(latencies):.2f} ms")
    print(f"Median / p50:  {statistics.median(latencies):.2f} ms")
    print(f"p95 latency:   {percentile(latencies, 95):.2f} ms")
    print(f"Minimum:       {min(latencies):.2f} ms")
    print(f"Maximum:       {max(latencies):.2f} ms")


if __name__ == "__main__":
    main()