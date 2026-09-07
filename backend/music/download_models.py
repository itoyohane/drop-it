"""Explicit model download; starting the API never downloads model weights."""

from huggingface_hub import snapshot_download

from backend.config import Settings


if __name__ == "__main__":
    settings = Settings()
    for model, revision in (
        (settings.description_model, settings.description_model_revision),
        (settings.text_embedding_model, settings.text_embedding_model_revision),
    ):
        snapshot_download(
            model, revision=revision, cache_dir=str(settings.data_dir / "models"),
            allow_patterns=["*.json", "*.txt", "*.model", "*.bin", "*.safetensors"],
        )
