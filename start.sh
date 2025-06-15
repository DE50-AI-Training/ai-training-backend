#!/bin/bash

# Démarrer Celery worker en arrière-plan
cd /app/srcAdd commentMore actions
celery -A trainer.trainer:app worker --loglevel=info &

# Démarrer FastAPI
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000