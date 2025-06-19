from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.
    """

    uvicorn_workers: int = Field(1, env="UVICORN_WORKERS")
    port: int = Field(8000, env="PORT")
    database_url: str = Field("sqlite:///database.db", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    storage_path: str = Field("storage", env="STORAGE_PATH")

    class Config:
        """
        Configuration for the Settings class.
        """ 
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
