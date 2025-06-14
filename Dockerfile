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

# Configure poetry to avoid virtual environments
ENV POETRY_VIRTUALENVS_CREATE=false

# Install dependencies (without ML libraries)
RUN poetry install --only=main --no-interaction --no-ansi

# Install PyTorch CPU-only version (lighter than full PyTorch)
RUN pip install torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install torchprofile
RUN pip install torchprofile==0.0.4

# Copy source code
COPY src/ src/

# Create necessary directories and set permissions
RUN mkdir -p storage/datasets storage/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Default command to run both API and celery worker
CMD ["sh", "-c", "celery -A trainer.trainer.app worker --loglevel=info --detach && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"]
