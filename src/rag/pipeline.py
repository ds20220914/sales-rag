"""
RAG pipeline: query → retrieval → prompt engineering → LLM → response.

Two operating modes
-------------------
direct  (default)
    The pipeline calls both retrieval tools unconditionally, builds a
    two-section context (summaries then transactions), and calls the LLM once.
    Predictable, fast, easy to debug.

agent
    The pipeline passes tool schemas to the LLM and runs an agentic loop:
    the model decides which tools to call, with what queries and filters,
    until it has enough context to answer.
    Requires a model with function/tool-calling support.

The pipeline is fully provider-agnostic: it receives a BaseLLMAdapter
instance and never imports OpenAI or Ollama directly.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import json

from llm.openai_adapter import OpenAIAdapter
from llm.ollama_adapter import OllamaAdapter
from vector_db.store import get_client, get_embedding_function, get_collection
from rag.tools import RetrievalTool, make_tools

_DEFAULT_DB = os.path.join(_SRC_DIR, "vector_db", "chroma_db")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_DIRECT = """\
You are a business intelligence analyst specialized in retail sales data.
You answer questions about the Superstore dataset (2014–2017, ~10,000 transactions).
Base your answers strictly on the provided context snippets.
Be concise, cite specific numbers, and clearly label years, categories, or regions.
If the context lacks sufficient data to answer, say so explicitly."""

_SYSTEM_AGENT = """\
You are a business intelligence analyst with access to a Superstore sales database (2014–2017).
Use the available tools to retrieve the data you need before answering.
Guidelines:
- Call search_summaries for aggregate statistics, trends, rankings, and comparisons.
- Call search_transactions for concrete order examples or to verify individual-level claims.
- You may call tools multiple times with different queries or filters.
- Once you have sufficient data, give a concise, number-backed answer."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_where(where) -> dict | None:
    """Sanitise a metadata filter dict coming from LLM arguments or the UI."""
    if where is None:
        return None
    if isinstance(where, str):
        try:
            where = json.loads(where)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(where, dict):
        return None

    if "$and" in where:
        clauses = where["$and"]
        if isinstance(clauses, list):
            seen: dict[str, set] = {}
            for clause in clauses:
                if isinstance(clause, dict):
                    for k, v in clause.items():
                        seen.setdefault(k, set()).add(str(v))
            if any(len(vals) > 1 for vals in seen.values()):
                return None

    return where


def _fmt_hits(hits: list[dict]) -> str:
    return "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)


def _build_direct_context(summary_hits: list[dict], txn_hits: list[dict]) -> str:
    parts: list[str] = []
    if summary_hits:
        parts.append("### Aggregated summaries\n" + _fmt_hits(summary_hits))
    if txn_hits:
        parts.append("### Transaction-level examples\n" + _fmt_hits(txn_hits))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline for Superstore sales analysis.

    Parameters
    ----------
    llm         : OpenAIAdapter or OllamaAdapter instance
    persist_dir : path to the ChromaDB directory built by build_index.py
    mode        : 'direct' or 'agent'
    n_summary   : default top-k for search_summaries  (direct mode)
    n_txn       : default top-k for search_transactions (direct mode)
    """

    def __init__(
        self,
        llm: OpenAIAdapter | OllamaAdapter,
        persist_dir: str = _DEFAULT_DB,
        mode: str = "direct",
        n_summary: int = 5,
        n_txn: int = 2,
    ) -> None:
        if mode not in ("direct", "agent"):
            raise ValueError(f"mode must be 'direct' or 'agent', got {mode!r}")

        self._llm = llm
        self.mode = mode
        self.n_summary = n_summary
        self.n_txn = n_txn

        chroma = get_client(persist_dir)
        ef = get_embedding_function()
        self.tools: dict[str, RetrievalTool] = make_tools(
            get_collection(chroma, "summaries",    ef),
            get_collection(chroma, "transactions", ef),
        )

        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Public retrieval helpers
    # ------------------------------------------------------------------

    def retrieve_summaries(
        self,
        query: str,
        where: dict | None = None,
        n_results: int | None = None,
    ) -> list[dict]:
        return self.tools["search_summaries"](
            query=query, where=where,
            n_results=n_results or self.n_summary,
        )

    def retrieve_transactions(
        self,
        query: str,
        where: dict | None = None,
        n_results: int | None = None,
    ) -> list[dict]:
        return self.tools["search_transactions"](
            query=query, where=where,
            n_results=n_results or self.n_txn,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        summary_where: dict | None = None,
        txn_where: dict | None = None,
        include_transactions: bool = True,
        use_memory: bool = True,
    ) -> dict:
        if self.mode == "direct":
            answer, summary_hits, txn_hits = self._run_direct(
                question, summary_where, txn_where, include_transactions, use_memory,
            )
        else:
            answer, summary_hits, txn_hits = self._run_agent(question, use_memory)

        return {
            "answer":       answer,
            "summary_hits": summary_hits,
            "txn_hits":     txn_hits,
            "mode":         self.mode,
        }

    # ------------------------------------------------------------------
    # Direct mode
    # ------------------------------------------------------------------

    def _run_direct(
        self,
        question: str,
        summary_where: dict | None,
        txn_where: dict | None,
        include_transactions: bool,
        use_memory: bool,
    ) -> tuple[str, list[dict], list[dict]]:
        summary_hits = self.retrieve_summaries(question, where=summary_where)
        txn_hits = (
            self.retrieve_transactions(question, where=txn_where)
            if include_transactions else []
        )
        context = _build_direct_context(summary_hits, txn_hits)

        messages: list[dict] = [{"role": "system", "content": _SYSTEM_DIRECT}]
        if use_memory and self.history:
            messages.extend(self.history[-6:])
        messages.append({
            "role":    "user",
            "content": f"Context (from Superstore sales database):\n{context}\n\nQuestion: {question}",
        })

        llm_msg = self._llm.chat(messages)
        answer = llm_msg.content or ""
        self._update_history(question, answer, use_memory)
        return answer, summary_hits, txn_hits

    # ------------------------------------------------------------------
    # Agent mode
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        question: str,
        use_memory: bool,
    ) -> tuple[str, list[dict], list[dict]]:
        if not self._llm.supports_tools():
            # Graceful degradation: fall back to direct mode
            return self._run_direct(question, None, None, True, use_memory)

        tool_schemas = [t.to_openai_schema() for t in self.tools.values()]
        summary_hits: list[dict] = []
        txn_hits:     list[dict] = []

        messages: list[dict] = [{"role": "system", "content": _SYSTEM_AGENT}]
        if use_memory and self.history:
            messages.extend(self.history[-6:])
        messages.append({"role": "user", "content": question})

        while True:
            llm_msg = self._llm.chat(messages, tools=tool_schemas)

            # No tool calls → final answer
            if not llm_msg.tool_calls:
                answer = llm_msg.content or ""
                self._update_history(question, answer, use_memory)
                return answer, summary_hits, txn_hits

            # Append assistant turn (keep raw tool_calls for providers that need it)
            assistant_turn: dict = {
                "role":    "assistant",
                "content": llm_msg.content,
            }
            # Re-attach raw tool_calls blob if the underlying SDK object is present
            # (OpenAI SDK needs the original object; Ollama works with dicts)
            if llm_msg.raw is not None and hasattr(llm_msg.raw, "tool_calls"):
                assistant_turn["tool_calls"] = llm_msg.raw.tool_calls
            else:
                assistant_turn["tool_calls"] = [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in llm_msg.tool_calls
                ]
            messages.append(assistant_turn)

            # Execute each tool call and feed results back
            for tc in llm_msg.tool_calls:
                if tc.name not in self.tools:
                    result_text = f"Error: unknown tool '{tc.name}'."
                else:
                    hits = self.tools[tc.name](
                        query=tc.arguments.get("query", ""),
                        where=_parse_where(tc.arguments.get("where")),
                        n_results=tc.arguments.get("n_results"),
                    )
                    result_text = _fmt_hits(hits) if hits else "No results found."
                    if tc.name == "search_summaries":
                        summary_hits.extend(hits)
                    else:
                        txn_hits.extend(hits)

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_text,
                })

    # ------------------------------------------------------------------
    # Conversation memory
    # ------------------------------------------------------------------

    def _update_history(self, question: str, answer: str, use_memory: bool) -> None:
        if use_memory:
            self.history.append({"role": "user",      "content": question})
            self.history.append({"role": "assistant",  "content": answer})

    def reset_memory(self) -> None:
        self.history.clear()