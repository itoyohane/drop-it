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
    intent_fallback_enabled: bool = Field(
        True, validation_alias="DROPIT_INTENT_FALLBACK_ENABLED"
    )
    ollama_base_url: str = Field(
        "http://127.0.0.1:11434/v1", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(
        "hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M", validation_alias="OLLAMA_MODEL"
    )
    ollama_timeout_seconds: float = Field(
        8.0, gt=0, le=120, validation_alias="OLLAMA_TIMEOUT_SECONDS"
    )
    agent_memory_max_messages: int = Field(
        100, ge=12, le=1000, validation_alias="DROPIT_AGENT_MEMORY_MAX_MESSAGES"
    )
    agent_memory_ttl_seconds: float = Field(
        1800, gt=0, le=86400, validation_alias="DROPIT_AGENT_MEMORY_TTL_SECONDS"
    )
    agent_context_window_tokens: int = Field(
        32768, ge=1024, le=2_000_000, validation_alias="DROPIT_AGENT_CONTEXT_WINDOW_TOKENS"
    )
    agent_context_compaction_ratio: float = Field(
        .8, ge=.5, le=.95, validation_alias="DROPIT_AGENT_CONTEXT_COMPACTION_RATIO"
    )
    agent_context_keep_messages: int = Field(
        6, ge=1, le=50, validation_alias="DROPIT_AGENT_CONTEXT_KEEP_MESSAGES"
    )
    agent_context_reserved_tokens: int = Field(
        4096, ge=0, le=131072, validation_alias="DROPIT_AGENT_CONTEXT_RESERVED_TOKENS"
    )
    agent_context_summary_tokens: int = Field(
        512, ge=64, le=4096, validation_alias="DROPIT_AGENT_CONTEXT_SUMMARY_TOKENS"
    )
    agent_run_lease_seconds: float = Field(
        15.0, gt=0.05, le=300, validation_alias="DROPIT_AGENT_RUN_LEASE_SECONDS"
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
    # Music RAG uses hosted APIs for description generation and embeddings.  Keep the
    # API keys separate so a deployment can rotate one provider without touching the
    # agent/chat credentials.
    description_model: str = Field("deepseek-v4.1-flash", validation_alias="DROPIT_DESCRIPTION_MODEL")
    description_model_revision: str = Field("api", validation_alias="DROPIT_DESCRIPTION_MODEL_REVISION")
    description_api_key: SecretStr | None = Field(
        None, validation_alias="DEEPSEEK_DESCRIPTION_API_KEY"
    )
    description_base_url: str = Field(
        "https://api.deepseek.com", validation_alias="DEEPSEEK_DESCRIPTION_BASE_URL"
    )
    description_timeout_seconds: float = Field(
        60.0, gt=0, le=300, validation_alias="DEEPSEEK_DESCRIPTION_TIMEOUT_SECONDS"
    )
    text_embedding_model: str = Field(
        "qwen3.7-text-embedding", validation_alias="DROPIT_TEXT_EMBEDDING_MODEL"
    )
    text_embedding_model_revision: str = Field(
        "api", validation_alias="DROPIT_TEXT_EMBEDDING_MODEL_REVISION"
    )
    text_embedding_dimensions: int = Field(
        1024, ge=1, le=4096, validation_alias="DROPIT_TEXT_EMBEDDING_DIMENSIONS"
    )
    dashscope_api_key: SecretStr | None = Field(
        None, validation_alias="DASHSCOPE_API_KEY"
    )
    dashscope_base_url: str = Field(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias="DASHSCOPE_BASE_URL",
    )
    dashscope_timeout_seconds: float = Field(
        60.0, gt=0, le=300, validation_alias="DASHSCOPE_TIMEOUT_SECONDS"
    )
    chroma_dir: Path | None = Field(
        None, validation_alias="DROPIT_CHROMA_DIR"
    )
    music_model_device: str = Field("cpu", validation_alias="DROPIT_MUSIC_MODEL_DEVICE")
    music_models_local_files_only: bool = Field(
        True, validation_alias="DROPIT_MUSIC_MODELS_LOCAL_FILES_ONLY"
    )
    log_level: str = Field("INFO", validation_alias="DROPIT_LOG_LEVEL")

    def configure_langsmith(self) -> None:
        """Make .env LangSmith settings available to LangChain's tracer."""
        if not self.langsmith_tracing:
            # An inherited/.env tracing flag can otherwise start LangSmith's
            # non-daemon control thread as soon as a graph invokes a model.
            # Explicitly disable both supported tracing switches when this
            # process is configured not to trace (notably test workers).
            os.environ["LANGSMITH_TRACING"] = "false"
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
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
    def music_description_api_key(self) -> SecretStr | None:
        """Use the dedicated description key, falling back to the agent key."""
        return self.description_api_key or self.deepseek_api_key

    @property
    def resolved_chroma_dir(self) -> Path:
        return self.chroma_dir or (self.data_dir / "chroma")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
