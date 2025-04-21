import os

import uvicorn
from fastapi import FastAPI

from config import settings

app = FastAPI()


def run_dev() -> None:
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=settings.port,
        workers=settings.uvicorn_workers,
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


@app.get("/")
async def root():
    return {"message": "Hello World"}
