FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml ./
COPY motionsense ./motionsense
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir .
COPY artifacts ./artifacts
ENV MOTIONSENSE_ARTIFACT_DIR=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "motionsense.api:app", "--host", "0.0.0.0", "--port", "8000"]
