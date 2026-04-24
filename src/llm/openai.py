"""
OpenAI-compatible provider — wraps the `openai` Python SDK.

Works with the official OpenAI API and any compatible endpoint:
  - OpenAI:      (default, no base_url needed)
  - Groq:        base_url="https://api.groq.com/openai/v1"
  - Together AI: base_url="https://api.together.xyz/v1"
  - LM Studio:   base_url="http://localhost:1234/v1"
  - Ollama REST: base_url="http://localhost:11434/v1"
"""

from __future__ import annotations

import json
from typing import Generator

from llm.base import ChatResponse, LLMProvider, ToolCall


class OpenAIProvider(LLMProvider):
    """
    Parameters
    ----------
    model    : model name as expected by the endpoint
    api_key  : API key (pass any non-empty string for local servers)
    base_url : override https://api.openai.com/v1 with a compatible endpoint
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        from openai import OpenAI  # lazy — not required when using Ollama
        self._client = OpenAI(api_key=api_key or None, base_url=base_url or None)
        self.model = model

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tcs: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                tcs.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        # Build a serialisable assistant dict (the SDK object is not JSON-serialisable)
        # content must be a string (not None) for some compatible APIs
        raw: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            raw["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return ChatResponse(content=msg.content, tool_calls=tcs, raw_message=raw)

    def stream_chat(self, messages: list[dict]) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=self.model, messages=messages, stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                yield text

    def make_tool_message(self, tool_call: ToolCall, content: str) -> dict:
        # OpenAI requires tool_call_id so the model can pair results with calls
        return {"role": "tool", "tool_call_id": tool_call.id, "content": content}