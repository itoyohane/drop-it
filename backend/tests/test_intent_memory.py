from backend.agent.intent import Intent, IntentRecognizer, IntentResult, OllamaIntentFallback
from backend.agent.memory import ContextCompressor, ShortTermMemory


def test_intent_recognizer_routes_core_music_requests():
    recognizer = IntentRecognizer()
    assert recognizer.recognize("帮我编排一个 45 分钟 warm-up set").name == Intent.GENERATE_SET
    assert recognizer.recognize("找几首和这首相似的歌").name == Intent.FIND_SIMILAR
    assert recognizer.recognize("曲库里有哪些 124 BPM 的歌").name == Intent.SEARCH_LIBRARY
    assert recognizer.recognize("解释一下 Camelot wheel").name == Intent.MUSIC_CHAT
    assert recognizer.recognize("帮我写一段 Python 代码").name == Intent.OVERSTEP
    assert recognizer.recognize("分析一下这只股票").name == Intent.OVERSTEP
    assert recognizer.recognize("聊聊最新政治新闻").name == Intent.OVERSTEP


def test_intent_recognizer_uses_fallback_only_after_rules_miss():
    class FakeFallback:
        def __init__(self):
            self.calls = []

        def classify(self, text):
            self.calls.append(text)
            return IntentResult(Intent.FIND_SIMILAR, 0.8, "fake")

    fallback = FakeFallback()
    recognizer = IntentRecognizer(fallback=fallback)

    assert recognizer.recognize("帮我找歌").name == Intent.SEARCH_LIBRARY
    numeric = recognizer.recognize("111")
    assert numeric.name == Intent.MUSIC_CHAT and not numeric.allows_tools
    assert recognizer.recognize("这首适合什么场景？").name == Intent.FIND_SIMILAR
    assert fallback.calls == ["这首适合什么场景？"]


def test_ollama_result_accepts_overstep_and_rejects_unknown_routes():
    result = OllamaIntentFallback._parse('{"intent":"overstep","confidence":0.91}')
    assert result.name == Intent.OVERSTEP and result.confidence == .91
    invalid = OllamaIntentFallback._parse('{"intent":"invented","confidence":1}')
    assert invalid.name == Intent.MUSIC_CHAT and invalid.confidence == 0
    assert not invalid.allows_tools and not result.allows_tools


def test_short_term_memory_is_bounded_isolated_and_expires():
    now = [0.0]
    memory = ShortTermMemory(max_messages=2, ttl_seconds=10, clock=lambda: now[0])
    memory.remember("a", "user", "one")
    memory.remember("a", "assistant", "two")
    memory.remember("a", "user", "three")
    memory.remember("b", "user", "other")
    assert [item["content"] for item in memory.messages("a")] == ["two", "three"]
    assert [item["content"] for item in memory.messages("b")] == ["other"]
    now[0] = 11
    assert memory.messages("a") == [] and memory.messages("b") == []


def test_context_compressor_triggers_at_eighty_percent_and_preserves_recent_messages():
    compressor = ContextCompressor(
        context_window_tokens=1024, trigger_ratio=.8, keep_messages=2, reserved_tokens=0
    )
    messages = [
        {"role": "user", "content": "a" * 1500},
        {"role": "assistant", "content": "b" * 1500},
        {"role": "user", "content": "keep-user"},
        {"role": "assistant", "content": "keep-assistant"},
    ]
    assert compressor.should_compact(messages)
    older, recent = compressor.split(messages)
    assert len(older) == 2
    assert [item["content"] for item in recent] == ["keep-user", "keep-assistant"]
    compacted = compressor.with_summary("facts only", recent)
    assert compacted[0]["content"].startswith("[历史摘要")
    assert compacted[-2:] == recent
