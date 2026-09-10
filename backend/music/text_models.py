"""Description and embedding clients used by the music RAG pipeline.

The legacy local model classes remain import-compatible for old callers, but the
production defaults are the hosted DeepSeek and DashScope clients below.
"""

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


class DeepSeekTrackDescriptor:
    """Generate a short retrieval document from measured librosa features.

    DeepSeek is only given deterministic features.  The prompt explicitly asks it
    not to infer artist, genre, instrumentation, or other facts that librosa did
    not measure, so the generated text remains suitable for semantic retrieval.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_key = (
            f"{settings.description_model}@{settings.description_model_revision}:features-v2"
        )

    @staticmethod
    def _facts(track: Track) -> str:
        details = track.analysis_details
        return (
            f"music audio; {track.bpm:.1f} BPM; key {track.key}; Camelot {track.camelot_key}; "
            f"energy {track.energy:.2f}; spectral centroid "
            f"{float(details.get('spectral_centroid_hz', 0)):.0f} Hz; onset strength "
            f"{float(details.get('onset_strength', 0)):.2f}"
        )

    @staticmethod
    def _endpoint(base_url: str) -> str:
        endpoint = base_url.rstrip("/")
        return endpoint if endpoint.endswith("/chat/completions") else f"{endpoint}/chat/completions"

    def describe(self, track: Track) -> str:
        import httpx

        api_key = self.settings.music_description_api_key
        if not api_key or not api_key.get_secret_value().strip():
            raise RuntimeError(
                "DeepSeek 歌曲描述 API key 未配置，请设置 DEEPSEEK_DESCRIPTION_API_KEY "
                "或 DEEPSEEK_API_KEY。"
            )
        facts = self._facts(track)
        response = httpx.post(
            self._endpoint(self.settings.description_base_url),
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value().strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.description_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You write concise music retrieval descriptions. Use only the measured "
                            "features supplied by the user. Never invent genre, vocals, instruments, "
                            "artist, era, or mood. Return one short English phrase."
                        ),
                    },
                    {"role": "user", "content": f"Measured features: {facts}"},
                ],
                "temperature": 0.1,
                "max_tokens": 96,
                "stream": False,
            },
            timeout=self.settings.description_timeout_seconds,
        )
        if response.is_error:
            raise RuntimeError(f"DeepSeek 歌曲描述请求失败（HTTP {response.status_code}）")
        try:
            payload = response.json()
            generated = payload["choices"][0]["message"]["content"]
            if isinstance(generated, list):
                generated = " ".join(
                    item.get("text", "") for item in generated if isinstance(item, dict)
                )
            generated = str(generated or "").strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("DeepSeek 歌曲描述响应格式无效") from exc
        return f"{facts}. {generated}" if generated else facts


class DashScopeTextEmbedder:
    """Call DashScope's OpenAI-compatible text embedding endpoint."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.dimensions = settings.text_embedding_dimensions
        self.model_key = (
            f"{settings.text_embedding_model}@{settings.text_embedding_model_revision}:dashscope-v1"
        )

    @staticmethod
    def _endpoint(base_url: str) -> str:
        endpoint = base_url.rstrip("/")
        return endpoint if endpoint.endswith("/embeddings") else f"{endpoint}/embeddings"

    def text(self, value: str) -> np.ndarray:
        import httpx

        if not value.strip():
            raise ValueError("音乐描述不能为空")
        api_key = self.settings.dashscope_api_key
        if not api_key or not api_key.get_secret_value().strip():
            raise RuntimeError("DashScope API key 未配置，请设置 DASHSCOPE_API_KEY。")
        response = httpx.post(
            self._endpoint(self.settings.dashscope_base_url),
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value().strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.text_embedding_model,
                "input": value,
                "dimensions": self.dimensions,
                "encoding_format": "float",
            },
            timeout=self.settings.dashscope_timeout_seconds,
        )
        if response.is_error:
            raise RuntimeError(f"DashScope 向量请求失败（HTTP {response.status_code}）")
        try:
            payload = response.json()
            rows = payload["data"]
            row = next(item for item in rows if int(item.get("index", 0)) == 0)
            vector = unit_vector(row["embedding"])
        except (KeyError, IndexError, StopIteration, TypeError, ValueError) as exc:
            raise RuntimeError("DashScope 向量响应格式无效") from exc
        if vector.shape != (self.dimensions,):
            raise ValueError(
                f"DashScope 向量维度为 {vector.size}，配置期望 {self.dimensions}。"
            )
        return vector


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
