from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from humps import camelize
from sqlmodel import SQLModel

from api.database import engine
from api.routers import datasets, models


class CamelCaseJSONResponse(JSONResponse):
    def render(self, content):
        if content:
            return super().render(camelize(content))
        return super().render(content)


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield


app = FastAPI(
    lifespan=lifespan, default_response_class=CamelCaseJSONResponse, docs_url="/"
)
app.include_router(models.router)
app.include_router(datasets.router)
