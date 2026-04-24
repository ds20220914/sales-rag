"""
OpenAI LLM adapter.

Wraps the openai Python SDK and translates responses into normalised
LLMMessage / ToolCall objects defined in base.py.

Supported models: gpt-4o, gpt-4o-mini, gpt-4-turbo, …
"""

from __future__ import annotations

import json
from openai import OpenAI

from .base import BaseLLMAdapter, LLMMessage, ToolCall


class OpenAIAdapter(BaseLLMAdapter):
    """
    Adapter for OpenAI chat-completion models.

    Parameters
    ----------
    model   : model name, e.g. 'gpt-4o-mini'
    api_key : OpenAI API key; falls back to OPENAI_API_KEY env var
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)   # None → reads OPENAI_API_KEY

    # ------------------------------------------------------------------
    # BaseLLMAdapter interface
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        return "openai"

    def supports_tools(self) -> bool:
        return True

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMMessage:
        kwargs: dict = dict(model=self.model, messages=messages)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        raw_msg = response.choices[0].message

        tool_calls: list[ToolCall] = []
        if raw_msg.tool_calls:
            for tc in raw_msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return LLMMessage(
            content=raw_msg.content,
            tool_calls=tool_calls,
            raw=raw_msg,
        )