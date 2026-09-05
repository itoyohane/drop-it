"""One shared CLAP space for audio indexing and text-to-music retrieval."""

from threading import RLock
from typing import Protocol

import numpy as np

from backend.config import Settings
from backend.music.essentia_analyzer import load_audio


def unit_vector(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("无效的音乐向量")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("音乐向量不能为零")
    return vector / norm


class MusicEmbedder(Protocol):
    model_key: str
    dimensions: int

    def audio(self, path: str) -> np.ndarray: ...
    def text(self, query: str) -> np.ndarray: ...


class ClapEmbedder:
    dimensions = 512

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_key = f"{settings.clap_model}@{settings.clap_revision}:center3x10-v1"
        self._model = None
        self._processor = None
        self._lock = RLock()

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import ClapModel, ClapProcessor

            options = {
                "revision": self.settings.clap_revision,
                "cache_dir": str(self.settings.data_dir / "models"),
                "local_files_only": self.settings.clap_local_files_only,
            }
            processor = ClapProcessor.from_pretrained(self.settings.clap_model, **options)
            model = ClapModel.from_pretrained(self.settings.clap_model, **options)
            model = model.to(self.settings.clap_device).eval()
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "CLAP 未就绪：安装音频依赖并运行 python -m backend.music.download_models。"
            ) from exc
        self._processor, self._model = processor, model

    def audio(self, path: str) -> np.ndarray:
        samples = load_audio(path, 48000)
        window = 48000 * 10
        starts = sorted({max(0, min(len(samples) - window, int(len(samples) * ratio) - window // 2))
                         for ratio in (0.2, 0.5, 0.8)})
        clips = [samples[start:start + window] for start in starts]
        with self._lock:
            self._load()
            import torch

            inputs = self._processor(audios=clips, sampling_rate=48000, return_tensors="pt")
            with torch.inference_mode():
                vectors = self._model.get_audio_features(**inputs.to(self.settings.clap_device))
            vectors = vectors.cpu().numpy()
        return unit_vector(np.mean([unit_vector(vector) for vector in vectors], axis=0))

    def text(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("音乐描述不能为空")
        with self._lock:
            self._load()
            import torch

            inputs = self._processor(text=[query], return_tensors="pt", padding=True,
                                     truncation=True, max_length=77)
            with torch.inference_mode():
                vector = self._model.get_text_features(**inputs.to(self.settings.clap_device))
            return unit_vector(vector[0].cpu().numpy())
