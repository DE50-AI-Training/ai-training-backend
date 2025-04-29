from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlmodel import SQLModel

from api.database import engine
from api.routers import datasets, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield


app = FastAPI(lifespan=lifespan, docs_url="/")
app.include_router(models.router)
app.include_router(datasets.router)
