"""
Ollama LLM adapter.

Uses the `ollama` Python SDK (pip install ollama).
Tool-calling is supported on models that declare it (llama3.2, mistral, …).
If the selected model does not support tools and tools are requested,
the adapter falls back gracefully to a plain text call.

Environment variables (all optional):
    OLLAMA_HOST  : base URL of the Ollama server (default: http://localhost:11434)
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import ollama as _ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

from .base import BaseLLMAdapter, LLMMessage, ToolCall

# Models known to support tool-calling via Ollama
_TOOL_CAPABLE_MODELS = {
    "llama3.2", "llama3.2:3b", "llama3.2:1b",
    "llama3.1", "llama3.1:8b", "llama3.1:70b",
    "mistral", "mistral-nemo",
    "qwen2.5", "qwen2.5:7b",
    "firefunction-v2",
    "command-r-plus",
}


def _model_supports_tools(model: str) -> bool:
    base = model.split(":")[0].lower()
    return model.lower() in _TOOL_CAPABLE_MODELS or base in _TOOL_CAPABLE_MODELS


class OllamaAdapter(BaseLLMAdapter):
    """
    Adapter for locally-running Ollama models.

    Parameters
    ----------
    model : Ollama model tag, e.g. 'llama3.2:3b'
    host  : Ollama server URL; falls back to OLLAMA_HOST env var,
            then 'http://localhost:11434'
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str | None = None,
    ) -> None:
        if not _OLLAMA_AVAILABLE:
            raise ImportError(
                "The 'ollama' package is required for OllamaAdapter. "
                "Install it with:  pip install ollama"
            )
        self.model = model
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = _ollama.Client(host=self._host)

    # ------------------------------------------------------------------
    # BaseLLMAdapter interface
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        return "ollama"

    def supports_tools(self) -> bool:
        return _model_supports_tools(self.model)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMMessage:
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages)

        # Only pass tools when the model actually supports them
        use_tools = tools and self.supports_tools()
        if use_tools:
            kwargs["tools"] = tools

        response = self._client.chat(**kwargs)
        raw_msg = response.message          # ollama.Message

        # ---- tool calls ------------------------------------------------
        tool_calls: list[ToolCall] = []
        if use_tools and getattr(raw_msg, "tool_calls", None):
            for tc in raw_msg.tool_calls:
                # ollama SDK: tc.function.name / tc.function.arguments (dict)
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                # Ollama doesn't provide a call id — synthesise one
                call_id = f"ollama_{tc.function.name}_{len(tool_calls)}"
                tool_calls.append(ToolCall(id=call_id, name=tc.function.name, arguments=args))

        return LLMMessage(
            content=raw_msg.content or None,
            tool_calls=tool_calls,
            raw=raw_msg,
        )