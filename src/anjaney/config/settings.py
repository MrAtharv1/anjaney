"""Configuration models and YAML loaders using Pydantic Settings."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    data_synthetic: Path = Path("data/synthetic")
    experiments: Path = Path("experiments")


class SeedsConfig(BaseModel):
    global_seed: int = 42
    split_seed: int = 1337
    permutation_seed: int = 2026


class ScreeningConfig(BaseModel):
    window_size: int = Field(default=5, ge=1)
    warmup_windows: int = Field(default=3, ge=1)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    phq8_threshold: int = Field(default=10, ge=0, le=24)
    alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    target_abstention: float = Field(default=0.15, ge=0.0, le=1.0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANJANEY_", env_nested_delimiter="__")

    paths: PathsConfig = Field(default_factory=PathsConfig)
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str = "configs/default.yaml") -> "Settings":
        path = Path(yaml_path)
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)


_settings_instance: Settings | None = None


def get_settings(config_path: Path | str = "configs/default.yaml") -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings.from_yaml(config_path)
    return _settings_instance
