"""Explicit model download; starting the API never downloads model weights."""

from huggingface_hub import snapshot_download

from backend.config import Settings


if __name__ == "__main__":
    settings = Settings()
    snapshot_download(
        settings.clap_model, revision=settings.clap_revision,
        cache_dir=str(settings.data_dir / "models"),
        allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"],
    )
