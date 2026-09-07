"""Single-file intent recognition for routing and prompting a DropIt Agent turn."""

from dataclasses import dataclass
from enum import StrEnum


class Intent(StrEnum):
    SEARCH_LIBRARY = "search_library"
    FIND_SIMILAR = "find_similar_tracks"
    GENERATE_SET = "generate_dj_set"
    MUSIC_CHAT = "music_chat"


@dataclass(frozen=True)
class IntentResult:
    name: Intent
    confidence: float
    guidance: str


class IntentRecognizer:
    """Cheap deterministic pre-router; the chat model still makes the final tool decision."""

    _set_terms = ("dj set", "set", "歌单", "编排", "排歌", "混音", "暖场", "开场", "峰值时段")
    _similar_terms = ("相似", "类似", "像这首", "接在后面", "下一首", "similar", "sounds like")
    _search_terms = ("找歌", "搜歌", "搜索", "曲库", "有哪些歌", "歌曲", "track", "library", "bpm", "调性")

    def recognize(self, text: str) -> IntentResult:
        normalized = " ".join(text.casefold().split())
        if any(term in normalized for term in self._set_terms):
            return IntentResult(Intent.GENERATE_SET, .92, "优先确认约束并调用 generate_dj_set。")
        if any(term in normalized for term in self._similar_terms):
            return IntentResult(Intent.FIND_SIMILAR, .9, "先确认参考 track_id，再调用 find_similar_tracks。")
        if any(term in normalized for term in self._search_terms):
            return IntentResult(Intent.SEARCH_LIBRARY, .86, "使用 search_library 获取当前曲库证据。")
        return IntentResult(Intent.MUSIC_CHAT, .55, "按音乐问答处理；只有涉及本地曲库时才调用工具。")
