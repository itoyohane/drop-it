"""One LangChain Agent owns the conversation and retrieves music through three tools."""

from collections.abc import AsyncIterator
import logging
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from backend.config import Settings
from backend.models import ChatMessage, Playlist, ToolEvent, ToolResult
from backend.repositories import DropItStore
from backend.agent.prompts import SYSTEM_PROMPT
from backend.agent.tools import DropItToolRegistry

logger = logging.getLogger(__name__)


MODEL_NOT_CONFIGURED_ERROR = "系统模型未配置，请联系管理员在服务器 .env 中配置 DEEPSEEK_API_KEY 后重试。"


class ModelNotConfiguredError(RuntimeError):
    pass


class DropItAgent:
    def __init__(self, store: DropItStore, registry: DropItToolRegistry, settings: Settings):
        self.store, self.registry, self.settings = store, registry, settings
        self.model_configured = settings.model_configured

    def require_model(self) -> None:
        if not self.model_configured:
            raise ModelNotConfiguredError(MODEL_NOT_CONFIGURED_ERROR)

    def _chat_model(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.settings.model_name, api_key=self.settings.deepseek_api_key,
                          base_url=self.settings.deepseek_base_url, temperature=0.2)

    async def stream_chat(self, project_id: str, conversation_id: str,
                          text: str) -> AsyncIterator[dict[str, Any]]:
        self.require_model()
        user = self.store.add_message(project_id, conversation_id, "user", text)
        yield {"type": "user_saved", "message": user.model_dump(mode="json")}
        yield {"type": "status", "label": "正在思考"}
        history = self.store.list_messages(project_id, conversation_id, limit=40)
        messages = [{"role": item.role, "content": item.content} for item in history]
        parts: list[str] = []
        final_text = ""
        tool_events: list[ToolEvent] = []
        playlist = None

        try:
            agent = create_agent(model=self._chat_model(), tools=self.registry.tools_for(project_id),
                                 system_prompt=SYSTEM_PROMPT)
            async for event in agent.astream_events(
                {"messages": messages}, config={"recursion_limit": 16}, version="v2"
            ):
                kind, name = event.get("event"), str(event.get("name") or "")
                if kind == "on_chat_model_stream":
                    chunk = self._message_text(event.get("data", {}).get("chunk", ""))
                    if chunk:
                        parts.append(chunk)
                        yield {"type": "token", "content": chunk}
                elif kind == "on_chat_model_end":
                    final_text = self._message_text(event.get("data", {}).get("output", ""))
                elif kind == "on_tool_start" and name in self.registry.names:
                    yield {"type": "tool", "name": name, "status": "running", "summary": "正在执行"}
                elif kind == "on_tool_end" and name in self.registry.names:
                    output = event.get("data", {}).get("output", "")
                    result = self._parse_tool_result(output)
                    record = ToolEvent(name=name, status="done" if result.ok else "failed",
                                       summary=result.summary)
                    tool_events.append(record)
                    yield {"type": "tool", **record.model_dump()}
                    if result.ok and result.data.get("playlist_id"):
                        candidate = self.store.get_playlist(str(result.data["playlist_id"]))
                        if candidate and candidate.project_id == project_id:
                            playlist = candidate
        except Exception as exc:
            logger.exception("agent_turn_failed")
            content = self._friendly_model_error(exc)
            tool_events.append(ToolEvent(name="model", status="failed", summary=content))
            yield {"type": "error", "detail": content}
        else:
            content = final_text.strip() or "".join(parts).strip() or "未收到有效回答，请重试。"
        message = self.store.add_message(project_id, conversation_id, "assistant", content, tool_events)
        yield {"type": "complete", "message": message.model_dump(mode="json"),
               "playlist": playlist.model_dump(mode="json") if playlist else None}

    async def chat(self, project_id: str, conversation_id: str,
                   text: str) -> tuple[ChatMessage, Playlist | None]:
        async for event in self.stream_chat(project_id, conversation_id, text):
            if event["type"] == "complete":
                return (ChatMessage.model_validate(event["message"]),
                        Playlist.model_validate(event["playlist"]) if event["playlist"] else None)
        raise RuntimeError("Agent 未返回最终消息")

    @staticmethod
    def _message_text(value: Any) -> str:
        content = getattr(value, "content", value)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(block.get("text", "") for block in content if isinstance(block, dict))
        return ""

    @classmethod
    def _parse_tool_result(cls, value: Any) -> ToolResult:
        try:
            return ToolResult.model_validate_json(cls._message_text(value))
        except ValueError:
            return ToolResult(ok=False, summary="工具返回了无效结果，请重试。")

    @staticmethod
    def _friendly_model_error(exc: Exception) -> str:
        message = str(exc).lower()
        if "401" in message or "api key" in message or "authentication" in message:
            return "模型鉴权失败，请检查 DEEPSEEK_API_KEY 后重试。"
        if "429" in message or "rate limit" in message:
            return "模型请求达到速率或额度限制，请稍后重试。"
        # SDK error strings can include request/credential details.
        return "模型调用失败，请检查服务端日志或稍后重试。"
