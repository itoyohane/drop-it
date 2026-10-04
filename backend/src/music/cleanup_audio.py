"""Preview or release legacy upload copies; run while the backend is stopped."""

import argparse
from pathlib import Path

from backend.audio import sha256_file
from backend.config import Settings
from backend.music.audio_cache import AudioCache
from backend.repositories import DropItStore


def cleanup_candidates(store: DropItStore, root: Path) -> list[Path]:
    """Include old duplicate uploads only if their hash has durable features."""
    root = root.resolve()
    protected = {track_id for job in store.pending_jobs() if job.payload.get("force")
                 for track_id in job.payload.get("track_ids", [])}
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        track = store.get_track_by_file_hash(sha256_file(path))
        if track and track.id not in protected and AudioCache.analyzed(track):
            candidates.append(path)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="清理已完成音频分析的上传缓存；请先停止后端。")
    parser.add_argument("--apply", action="store_true", help="执行删除；默认仅预览")
    args = parser.parse_args()
    settings = Settings()
    if not (settings.data_dir / "dropit.db").is_file():
        parser.error("数据目录中没有 dropit.db")
    store = DropItStore(str(settings.data_dir / "dropit.db"),
                        vector_store_path=str(settings.resolved_chroma_dir))
    try:
        candidates = cleanup_candidates(store, settings.data_dir / "imports" / "sources")
        total = sum(path.stat().st_size for path in candidates)
        print(f"{'清理' if args.apply else '可清理'} {len(candidates)} 个缓存文件，{total / 1024**2:.1f} MiB")
        for path in candidates:
            if args.apply:
                path.unlink(missing_ok=True)
            print(path)
    finally:
        store.close()


if __name__ == "__main__":
    main()
