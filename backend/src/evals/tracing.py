"""Provider-reported model usage; no invented token or currency estimates."""

from langchain_core.callbacks import BaseCallbackHandler


class ModelUsageRecorder(BaseCallbackHandler):
    def __init__(self):
        self.calls = 0
        self.calls_with_usage = 0
        self._totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        usage = None
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if not usage:
                    usage = (getattr(message, "response_metadata", {}) or {}).get("token_usage")
                if usage:
                    break
            if usage:
                break
        if not usage:
            usage = (response.llm_output or {}).get("token_usage")
        if not usage:
            return
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        if input_tokens is None or output_tokens is None:
            return
        self.calls_with_usage += 1
        self._totals["input_tokens"] += int(input_tokens)
        self._totals["output_tokens"] += int(output_tokens)
        self._totals["total_tokens"] += int(usage.get("total_tokens", input_tokens + output_tokens))

    @property
    def usage(self):
        return dict(self._totals) if self.calls_with_usage else None

    @property
    def summary(self):
        return {"model_calls": self.calls, "calls_with_usage": self.calls_with_usage,
                "status": ("unavailable" if not self.calls_with_usage else
                           "available" if self.calls == self.calls_with_usage else "partial")}
