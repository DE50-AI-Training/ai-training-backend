from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    uvicorn_workers: int = Field(1, env="UVICORN_WORKERS")
    port: int = Field(8000, env="PORT")
    database_url: str = Field("sqlite:///database.db", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
