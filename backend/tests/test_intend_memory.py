from backend.agent.intend import Intent, IntentRecognizer
from backend.agent.memory import ShortTermMemory


def test_intent_recognizer_routes_core_music_requests():
    recognizer = IntentRecognizer()
    assert recognizer.recognize("帮我编排一个 45 分钟 warm-up set").name == Intent.GENERATE_SET
    assert recognizer.recognize("找几首和这首相似的歌").name == Intent.FIND_SIMILAR
    assert recognizer.recognize("曲库里有哪些 124 BPM 的歌").name == Intent.SEARCH_LIBRARY
    assert recognizer.recognize("解释一下 Camelot wheel").name == Intent.MUSIC_CHAT


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
