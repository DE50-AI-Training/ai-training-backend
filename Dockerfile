# Use Python 3.11 as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.7.1

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure poetry: Don't create virtual env (we're in container), don't ask for confirmation
RUN poetry config virtualenvs.create false \
    && poetry config virtualenvs.in-project false

# Install dependencies
RUN poetry install --only=main --no-interaction --no-ansi

# Copy source code
COPY src/ src/
COPY storage/ storage/
COPY README.md .

# Create necessary directories and set permissions
RUN mkdir -p storage/datasets storage/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Default command to run both API and celery worker
CMD ["sh", "-c", "celery -A trainer.trainer.app worker --loglevel=info --detach && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"]
