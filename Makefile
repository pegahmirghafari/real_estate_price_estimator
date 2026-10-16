SHELL := /bin/bash
COMPOSE := docker compose -f docker-compose.test.yml

.PHONY: test-build test-unit test-integration benchmark test-all clean help

# Build the test Docker image
test-build:
	docker build -f Dockerfile.test -t ml-api-test .

# Run unit tests
test-unit: test-build
	docker run --rm \
		-v "$(CURDIR)/test-results:/app/test-results" \
		ml-api-test \
		pytest test/unit -v

# Run integration tests against the shared API container
test-integration:
	@set -e; \
	trap '$(COMPOSE) down --remove-orphans' EXIT; \
	$(COMPOSE) up -d --build --wait api; \
	$(COMPOSE) run --rm --no-deps test pytest test/integration -v

# Run both benchmarks against the same API container
benchmark:
	@set -e; \
	trap '$(COMPOSE) down --remove-orphans' EXIT; \
	$(COMPOSE) up -d --build --wait api; \
	echo ""; \
	echo "========================================"; \
	echo "API PERFORMANCE BENCHMARK"; \
	echo "========================================"; \
	$(COMPOSE) run --rm --no-deps test \
		python /app/benchmark.py; \
	echo ""; \
	echo "========================================"; \
	echo "RESOURCE CACHING BENCHMARK"; \
	echo "========================================"; \
	$(COMPOSE) run --rm --no-deps test \
		python /app/benchmark_loading.py

# Run integration tests, unit tests, and both benchmarks
test-all: test-build
	@set -e; \
	trap '$(COMPOSE) down --remove-orphans' EXIT; \
	echo ""; \
	echo "========================================"; \
	echo "INTEGRATION TESTS"; \
	echo "========================================"; \
	$(COMPOSE) up -d --build --wait api; \
	$(COMPOSE) run --rm --no-deps test pytest test/integration -v; \
	echo ""; \
	echo "========================================"; \
	echo "UNIT TESTS"; \
	echo "========================================"; \
	docker run --rm \
		-v "$(CURDIR)/test-results:/app/test-results" \
		ml-api-test \
		pytest test/unit -v; \
	echo ""; \
	echo "========================================"; \
	echo "API PERFORMANCE BENCHMARK"; \
	echo "========================================"; \
	$(COMPOSE) run --rm --no-deps test \
		python /app/benchmark.py; \
	echo ""; \
	echo "========================================"; \
	echo "RESOURCE CACHING BENCHMARK"; \
	echo "========================================"; \
	$(COMPOSE) run --rm --no-deps test \
		python /app/benchmark_loading.py; \
	echo ""; \
	echo "========================================"; \
	echo "ALL TESTS AND BENCHMARKS COMPLETED"; \
	echo "========================================"

# Remove test containers and generated results
clean:
	$(COMPOSE) down --remove-orphans
	rm -rf test-results/*

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make test-unit         Run unit tests"
	@echo "  make test-integration  Run integration tests"
	@echo "  make benchmark         Run API and caching benchmarks"
	@echo "  make test-all          Run all tests and benchmarks"
	@echo "  make clean             Remove test containers and results"