FROM python:3.11-slim

WORKDIR /app

# Install only API dependencies (no jupyter/notebook bloat)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy artifacts and API code
COPY model_artifacts/ ./model_artifacts/
COPY WA_Fn-UseC_-HR-Employee-Attrition.csv .
COPY api/ ./api/

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
