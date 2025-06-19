from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel

from api.database import engine
from api.routers import datasets, models


# Database initialization on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield


# Start the FastAPI app
app = FastAPI(
    lifespan=lifespan,
    docs_url="/",
    title="AI Training Backend API",
    description="""\
## API for Machine Learning Model Training

This API manages the complete ML model lifecycle:
- **Upload and automatic analysis** of CSV datasets
- **Creation and configuration** of MLP (Multi-Layer Perceptron) models
- **Asynchronous training** with real-time monitoring
- **Inference** on complete datasets or individual observations

### Main Features

- **Complete CSV support**: Automatic delimiter and type detection
- **Flexible architectures**: Free configuration of MLP layers
- **Distributed training**: Via Celery with real-time metrics
- **Two problem types**: Classification and regression
- **Complete REST API**: Full CRUD + specialized operations

### Technologies

- **FastAPI**: Modern and performant web framework
- **SQLModel**: Type-safe ORM for persistence
- **Redis**: Cache and task coordination
- **Celery**: Asynchronous training execution
- **PyTorch**: Machine learning backend

### Typical Workflow

1. **Upload dataset** → `POST /datasets/` 

2. **Finalize dataset** → `POST /datasets/{id}`

3. **Create model** → `POST /models/`

4. **Start training** → `POST /models/{id}/train`

5. **Monitor progress** → `GET /models/trainings`

6. **Make predictions** → `POST /models/{id}/infer-single`
    """,
    version="1.0.0",
    contact={
        "name": "DE50 Project Team",
        "url": "https://github.com/your-repo/ai-training-backend",
    },
    license_info={
        "name": "MIT",
    },
)

# Include API routers
app.include_router(models.router)
app.include_router(datasets.router)

# CORS configuration for frontend integration
origins = [
    "http://localhost:3000",  # React development server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
