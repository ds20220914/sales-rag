"""
llm — provider-agnostic LLM adapter layer.

Public API
----------
make_adapter(provider, model, **kwargs) -> BaseLLMAdapter
    Factory function used by app.py and tests.

Classes
-------
BaseLLMAdapter   — abstract interface
LLMMessage       — normalised response (content + tool_calls)
ToolCall         — normalised tool call
OpenAIAdapter    — OpenAI implementation
OllamaAdapter    — Ollama implementation
"""

from .base import BaseLLMAdapter, LLMMessage, ToolCall
from .openai_adapter import OpenAIAdapter
from .ollama_adapter import OllamaAdapter

__all__ = [
    "BaseLLMAdapter",
    "LLMMessage",
    "ToolCall",
    "OpenAIAdapter",
    "OllamaAdapter",
    "make_adapter",
]


def make_adapter(
    provider: str,
    model: str,
    api_key: str | None = None,
    ollama_host: str | None = None,
) -> BaseLLMAdapter:
    """
    Factory that returns the correct adapter for *provider*.

    Parameters
    ----------
    provider    : 'openai' or 'ollama'
    model       : model name/tag passed through to the adapter
    api_key     : OpenAI API key (ignored for Ollama)
    ollama_host : Ollama server URL (ignored for OpenAI)

    Raises
    ------
    ValueError  if *provider* is not recognised
    """
    p = provider.lower()
    if p == "openai":
        return OpenAIAdapter(model=model, api_key=api_key)
    if p == "ollama":
        return OllamaAdapter(model=model, host=ollama_host)
    raise ValueError(
        f"Unknown LLM provider {provider!r}. Choose 'openai' or 'ollama'."
    )