from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel

from api.database import engine
from api.routers import datasets, models
from config import settings


# Create the database engine
@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield


# Start the FastAPI app
app = FastAPI(lifespan=lifespan, docs_url="/")
app.include_router(models.router)
app.include_router(datasets.router)
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize redis
redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
