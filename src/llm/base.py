"""
Abstract base class for LLM adapters.

Every adapter must implement:
    chat()   — single-turn or multi-turn completion
    supports_tools() — whether the model can use tool-calling
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Normalised representation of a single tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMMessage:
    """
    Normalised LLM response, regardless of provider.

    Attributes
    ----------
    content      : text content (may be None when the model only calls tools)
    tool_calls   : list of ToolCall objects (empty when model gives a text answer)
    raw          : the original provider response object, for debugging
    """
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = field(default=None, repr=False)


class BaseLLMAdapter(ABC):
    """
    Provider-agnostic interface that RAGPipeline talks to.

    Subclasses: OpenAIAdapter, OllamaAdapter
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMMessage:
        """
        Send *messages* to the model and return a normalised LLMMessage.

        Parameters
        ----------
        messages : OpenAI-style list of dicts
                   [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}]
        tools    : optional list of OpenAI-style tool schemas
                   (adapters that don't support tools should raise if called with them)
        """

    @abstractmethod
    def supports_tools(self) -> bool:
        """Return True if this adapter/model supports tool-calling."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Human-readable provider name, e.g. 'openai' or 'ollama'."""