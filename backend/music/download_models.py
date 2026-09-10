"""Validate hosted music-RAG configuration.

The default description and embedding models are API-backed, so there are no
Hugging Face weights to download at startup or via this command.
"""

from backend.config import Settings


if __name__ == "__main__":
    settings = Settings()
    print(f"DeepSeek description model: {settings.description_model}")
    print(f"DashScope embedding model: {settings.text_embedding_model}")
    print(f"Chroma directory: {settings.resolved_chroma_dir}")
    print("Hosted model weights are not downloaded locally; configure API keys in .env.")
