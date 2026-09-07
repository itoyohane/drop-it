from pathlib import Path
from typing import Literal
import logging
import os
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("dropit")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = Field(
        "development", validation_alias="DROPIT_ENV"
    )
    data_dir: Path = Field(Path("data"), validation_alias="DROPIT_DATA_DIR")
    model_name: str = Field("deepseek-v4-pro", validation_alias="DROPIT_MODEL")
    deepseek_api_key: SecretStr | None = Field(None, validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        "https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL"
    )
    langsmith_tracing: bool = Field(False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(
        None, validation_alias="LANGSMITH_API_KEY"
    )
    langsmith_project: str = Field(
        "drop-it-dev", validation_alias="LANGSMITH_PROJECT"
    )
    langsmith_endpoint: str = Field(
        "https://api.smith.langchain.com",
        validation_alias="LANGSMITH_ENDPOINT",
    )
    cors_origins: str = Field(
        "http://127.0.0.1:5173,http://localhost:5173", validation_alias="DROPIT_CORS_ORIGINS"
    )
    max_upload_mb: int = Field(1024, ge=10, le=4096, validation_alias="DROPIT_MAX_UPLOAD_MB")
    max_upload_files: int = Field(500, ge=1, le=5000, validation_alias="DROPIT_MAX_UPLOAD_FILES")
    description_model: str = Field("google/flan-t5-small", validation_alias="DROPIT_DESCRIPTION_MODEL")
    description_model_revision: str = Field("main", validation_alias="DROPIT_DESCRIPTION_MODEL_REVISION")
    text_embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", validation_alias="DROPIT_TEXT_EMBEDDING_MODEL"
    )
    text_embedding_model_revision: str = Field(
        "main", validation_alias="DROPIT_TEXT_EMBEDDING_MODEL_REVISION"
    )
    text_embedding_dimensions: int = Field(
        384, ge=1, le=4096, validation_alias="DROPIT_TEXT_EMBEDDING_DIMENSIONS"
    )
    music_model_device: str = Field("cpu", validation_alias="DROPIT_MUSIC_MODEL_DEVICE")
    music_models_local_files_only: bool = Field(
        True, validation_alias="DROPIT_MUSIC_MODELS_LOCAL_FILES_ONLY"
    )
    log_level: str = Field("INFO", validation_alias="DROPIT_LOG_LEVEL")

    def configure_langsmith(self) -> None:
        """Make .env LangSmith settings available to LangChain's tracer."""
        if not self.langsmith_tracing:
            return

        api_key = (
            self.langsmith_api_key.get_secret_value().strip()
            if self.langsmith_api_key
            else ""
        )
        if not api_key:
            logger.warning(
                "LangSmith tracing is enabled but LANGSMITH_API_KEY is missing"
            )
            return

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint

    @property
    def model_configured(self) -> bool:
        return bool(self.deepseek_api_key and self.deepseek_api_key.get_secret_value().strip())

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
