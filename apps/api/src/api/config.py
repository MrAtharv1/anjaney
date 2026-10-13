"""API configuration settings loading from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANJANEY_API_")

    app_name: str = "ANJANEY Campus Wellness API"
    app_version: str = "1.0.0"
    database_url: str = "sqlite+aiosqlite:///./anjaney_campus.db"
    bundle_path: Path = Path("data/processed/bundle/bundle.json")
    experiments_path: Path = Path("experiments")
    jwt_secret: str = "demo-super-secret-key-change-in-production-12345"
    jwt_algorithm: str = "HS256"
    k_anonymity_min: int = 5
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


settings = APISettings()
