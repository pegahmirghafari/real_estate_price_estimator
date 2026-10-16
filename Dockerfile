FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Create an unprivileged application user.
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime application code.
COPY src/main.py ./src/main.py
COPY src/config.py ./src/config.py
COPY src/api ./src/api
COPY src/utils ./src/utils

# Copy only runtime ML artifacts.
COPY src/model/model.pkl ./src/model/model.pkl
COPY src/model/imputer.pkl ./src/model/imputer.pkl
COPY src/model/model_features.json ./src/model/model_features.json

# Only this dataset is required during inference.
COPY src/data/zipcode_demographics.csv ./src/data/zipcode_demographics.csv

# Application files should belong to the non-root user.
RUN chown -R appuser:appgroup /app
WORKDIR /app/src
USER appuser
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]