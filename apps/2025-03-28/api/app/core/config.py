"""Settings module."""
from typing import List, Union
from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    # Project settings
    PROJECT_NAME: str = "Investment Recommendation System"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "0.1.0"
    SECRET_KEY: str = "your-secret-key"  # Change this in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    BACKEND_CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = [
        "http://localhost:3000",  # React frontend
        "http://localhost:8000",  # FastAPI backend
    ]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Validate CORS origins."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"  # Change this in production
    POSTGRES_DB: str = "investment_db"
    SQLALCHEMY_DATABASE_URI: str = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}/{POSTGRES_DB}"

    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # ML Model settings
    MODEL_PATH: str = "models/trained"
    SENTIMENT_MODEL: str = "models/sentiment"
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001

    # API Keys
    ALPHA_VANTAGE_KEY: str = "your-key-here"
    NEWS_API_KEY: str = "your-key-here"
    FRED_API_KEY: str = "your-key-here"

    # JWT Configuration
    ALGORITHM: str = "HS256"

    class Config:
        """Pydantic config."""
        case_sensitive = True
        env_file = ".env"

settings = Settings()
