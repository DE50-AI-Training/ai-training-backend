from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI
from sqlmodel import Session, SQLModel, create_engine

from api.routers import datasets, models
from config import settings


def run_dev() -> None:
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=settings.port,
        reload=True,
        factory=False,
    )


def run_prod() -> None:
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=settings.port,
        workers=settings.uvicorn_workers,
        reload=False,
        factory=False,
    )


connect_args = {"check_same_thread": False}
engine = create_engine(settings.database_url, connect_args=connect_args)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(models.router)
app.include_router(datasets.router)
