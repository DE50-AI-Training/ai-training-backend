import uvicorn
from sqlmodel import SQLModel

from api.database import engine
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


def create_db() -> None:
    SQLModel.metadata.create_all(engine)
