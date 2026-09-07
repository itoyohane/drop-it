"""Small local text-generation and embedding models for description-based music RAG."""

from threading import RLock
from typing import Protocol

import numpy as np

from backend.config import Settings
from backend.models import Track


def unit_vector(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("无效的文本向量")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("文本向量不能为零")
    return vector / norm


class TrackDescriptor(Protocol):
    model_key: str

    def describe(self, track: Track) -> str: ...


class TextEmbedder(Protocol):
    model_key: str
    dimensions: int

    def text(self, value: str) -> np.ndarray: ...


class SmallTextDescriptor:
    """Turn measured librosa features into a concise English retrieval document."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_key = f"{settings.description_model}@{settings.description_model_revision}:features-v1"
        self._model = None
        self._tokenizer = None
        self._lock = RLock()

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            options = {
                "revision": self.settings.description_model_revision,
                "cache_dir": str(self.settings.data_dir / "models"),
                "local_files_only": self.settings.music_models_local_files_only,
            }
            self._tokenizer = AutoTokenizer.from_pretrained(self.settings.description_model, **options)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.settings.description_model, **options
            ).to(self.settings.music_model_device).eval()
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "歌曲描述模型未就绪：请安装音频依赖并运行 python -m backend.music.download_models。"
            ) from exc

    def describe(self, track: Track) -> str:
        details = track.analysis_details
        facts = (
            f"music audio; {track.bpm:.1f} BPM; {track.key}; Camelot {track.camelot_key}; "
            f"energy {track.energy:.2f}; spectral centroid "
            f"{float(details.get('spectral_centroid_hz', 0)):.0f} Hz; onset strength "
            f"{float(details.get('onset_strength', 0)):.2f}"
        )
        prompt = (
            "Describe the sonic character of a music track in one short English phrase for semantic search. "
            "Use only these measured features; do not invent genre, vocals, instruments, era, or artist. "
            f"Features: {facts}"
        )
        with self._lock:
            self._load()
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {key: value.to(self.settings.music_model_device) for key, value in inputs.items()}
            with torch.inference_mode():
                tokens = self._model.generate(**inputs, max_new_tokens=48, do_sample=False)
            generated = self._tokenizer.decode(tokens[0], skip_special_tokens=True).strip()
        return f"{facts}. {generated}" if generated else facts


class SentenceTransformerEmbedder:
    """Embed generated track descriptions and user queries into one small text space."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.dimensions = settings.text_embedding_dimensions
        self.model_key = f"{settings.text_embedding_model}@{settings.text_embedding_model_revision}:description-v1"
        self._model = None
        self._lock = RLock()

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.settings.text_embedding_model,
                revision=self.settings.text_embedding_model_revision,
                cache_folder=str(self.settings.data_dir / "models"),
                local_files_only=self.settings.music_models_local_files_only,
                device=self.settings.music_model_device,
            )
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "文本向量模型未就绪：请安装音频依赖并运行 python -m backend.music.download_models。"
            ) from exc
        actual = int(self._model.get_sentence_embedding_dimension())
        if actual != self.dimensions:
            raise RuntimeError(f"文本向量模型维度为 {actual}，配置期望 {self.dimensions}。")

    def text(self, value: str) -> np.ndarray:
        if not value.strip():
            raise ValueError("音乐描述不能为空")
        with self._lock:
            self._load()
            vector = self._model.encode(value, convert_to_numpy=True, normalize_embeddings=True)
        return unit_vector(vector)
