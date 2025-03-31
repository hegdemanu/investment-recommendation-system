import secrets
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")
    
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    @field_validator("BACKEND_CORS_ORIGINS")
    def validate_urls(cls, urls: List[str]) -> List[str]:
        for url in urls:
            # Simple validation or use urllib.parse
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL: {url}")
        return urls

    PROJECT_NAME: str = "Investment Recommendation System"
    
    # Database configuration
    # For development, use SQLite
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///./investment_recommendation.db"
    
    # For production, use PostgreSQL
    # POSTGRES_SERVER: str = "localhost"
    # POSTGRES_USER: str = "postgres"
    # POSTGRES_PASSWORD: str = "postgres"
    # POSTGRES_DB: str = "investment_recommendation"
    # SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None
    #
    # @field_validator("SQLALCHEMY_DATABASE_URI", mode="before")
    # @classmethod
    # def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
    #     if isinstance(v, str):
    #         return v
    #     return PostgresDsn.build(
    #         scheme="postgresql",
    #         username=values.data.get("POSTGRES_USER"),
    #         password=values.data.get("POSTGRES_PASSWORD"),
    #         host=values.data.get("POSTGRES_SERVER"),
    #         path=f"{values.data.get('POSTGRES_DB') or ''}",
    #     )

    # Alpha Vantage API key
    ALPHA_VANTAGE_API_KEY: str = ""
    
    # News API key
    NEWS_API_KEY: str = ""
    
    # Redis configuration
    REDIS_HOST: Optional[str] = None
    REDIS_PORT: Optional[int] = None
    
    # Model paths
    MODEL_PATH: Optional[str] = None
    SENTIMENT_MODEL: Optional[str] = None
    
    # Training parameters
    BATCH_SIZE: Optional[int] = None
    LEARNING_RATE: Optional[float] = None
    
    # API Keys
    ALPHA_VANTAGE_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None


settings = Settings() 