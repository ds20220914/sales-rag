"""
RAG pipeline: query → retrieval → prompt engineering → LLM → response.

Two operating modes
-------------------
direct  (default)
    Retrieves summaries + transactions unconditionally, builds context, calls
    the LLM once.  Predictable, fast, easy to debug.

agent
    Passes tool schemas to the LLM and runs an agentic loop: the model decides
    which tools to call and with what filters until it has enough context.
    Requires a model with tool-calling support (llama3.2, mistral, gpt-4o, …).

LLM providers
-------------
Pass provider='ollama' (default) or provider='openai' together with any
keyword arguments accepted by make_llm() (model, api_key, base_url, …).

Usage examples
--------------
    RAGPipeline()                              # reads all settings from .env
    RAGPipeline(provider='ollama', model='mistral', mode='agent')
    RAGPipeline(provider='openai', model='gpt-4o-mini', api_key='sk-...')
"""

from __future__ import annotations

import json
import os
import sys
from typing import Generator

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
_ROOT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

from vector_db.store import get_client, get_embedding_function, get_collection
from rag.tools import make_tools
from llm import LLMProvider, make_llm

# ---------------------------------------------------------------------------
# Module-level defaults from .env
# ---------------------------------------------------------------------------

_DEFAULT_DB       = os.environ.get("CHROMA_DB_PATH",
                                   os.path.join(_SRC_DIR, "vector_db", "chroma_db"))
_DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
_DEFAULT_MODEL    = (
    os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if _DEFAULT_PROVIDER == "openai"
    else os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
)

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
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_where(where) -> dict | None:
    """
    Normalise a metadata filter coming from an LLM tool call.

    The LLM sometimes serialises the filter as a JSON string.  Also drops
    contradictory $and filters (same key, two different values) that would
    always return empty results.
    """
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


def _build_context(summary_hits: list[dict], txn_hits: list[dict]) -> str:
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
    persist_dir : ChromaDB directory (default: CHROMA_DB_PATH from .env)
    provider    : 'ollama' or 'openai'  (default: LLM_PROVIDER from .env)
    model       : model name            (default: OLLAMA_MODEL / OPENAI_MODEL from .env)
    mode        : 'direct' or 'agent'
    n_summary   : top-k for search_summaries   (direct / stream mode)
    n_txn       : top-k for search_transactions (direct / stream mode)
    **llm_kwargs: forwarded to make_llm() — e.g. api_key, base_url, host
    """

    def __init__(
        self,
        persist_dir: str = _DEFAULT_DB,
        provider:    str = _DEFAULT_PROVIDER,
        model:       str = _DEFAULT_MODEL,
        mode:        str = "direct",
        n_summary:   int = 5,
        n_txn:       int = 2,
        **llm_kwargs,
    ):
        if mode not in ("direct", "agent"):
            raise ValueError(f"mode must be 'direct' or 'agent', got {mode!r}")

        self.provider  = provider
        self.model     = model
        self.mode      = mode
        self.n_summary = n_summary
        self.n_txn     = n_txn

        # Inject credentials from env so callers don't need to pass them
        if provider == "openai":
            llm_kwargs.setdefault("api_key",  os.environ.get("OPENAI_API_KEY") or None)
            llm_kwargs.setdefault("base_url", os.environ.get("OPENAI_BASE_URL") or None)
        elif provider == "ollama":
            llm_kwargs.setdefault("host", os.environ.get("OLLAMA_HOST") or None)

        self.llm: LLMProvider = make_llm(provider, model=model, **llm_kwargs)

        client = get_client(persist_dir)
        ef     = get_embedding_function()
        self.tools = make_tools(
            get_collection(client, "summaries",    ef),
            get_collection(client, "transactions", ef),
        )

        self.history: list[dict] = []
        self._last_summary_hits: list[dict] = []
        self._last_txn_hits:     list[dict] = []

    @property
    def last_summary_hits(self) -> list[dict]:
        return self._last_summary_hits

    @property
    def last_txn_hits(self) -> list[dict]:
        return self._last_txn_hits

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
    # Public entry points
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        summary_where:        dict | None = None,
        txn_where:            dict | None = None,
        include_transactions: bool        = True,
        use_memory:           bool        = True,
    ) -> dict:
        """
        Single-shot answer (no streaming).

        Returns
        -------
        dict with keys: answer, summary_hits, txn_hits, mode
        """
        if self.mode == "direct":
            answer, summary_hits, txn_hits = self._run_direct(
                question, summary_where, txn_where, include_transactions, use_memory,
            )
        else:
            answer, summary_hits, txn_hits = self._run_agent(question, use_memory)

        self._last_summary_hits = summary_hits
        self._last_txn_hits     = txn_hits
        return {"answer": answer, "summary_hits": summary_hits,
                "txn_hits": txn_hits, "mode": self.mode}

    def stream(
        self,
        question: str,
        summary_where:        dict | None = None,
        txn_where:            dict | None = None,
        include_transactions: bool        = True,
        use_memory:           bool        = True,
    ) -> Generator[str, None, None]:
        """
        Streaming answer — yields text chunks as they arrive.

        After the generator is exhausted, retrieved chunks are accessible
        via last_summary_hits and last_txn_hits.

        Agent mode does not support true streaming (tool calls must complete
        before a response can be formed); it falls back to ask() and yields
        the full answer as a single chunk.
        """
        if self.mode == "agent":
            result = self.ask(question, use_memory=use_memory)
            yield result["answer"]
            return

        messages, summary_hits, txn_hits = self._prepare_direct(
            question, summary_where, txn_where, include_transactions, use_memory,
        )
        self._last_summary_hits = summary_hits
        self._last_txn_hits     = txn_hits

        chunks: list[str] = []
        for chunk in self.llm.stream_chat(messages):
            chunks.append(chunk)
            yield chunk

        self._update_history(question, "".join(chunks), use_memory)

    def reset_memory(self) -> None:
        self.history.clear()

    # ------------------------------------------------------------------
    # Internal — direct mode
    # ------------------------------------------------------------------

    def _prepare_direct(
        self,
        question:             str,
        summary_where:        dict | None,
        txn_where:            dict | None,
        include_transactions: bool,
        use_memory:           bool,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Retrieve hits and build the message list for direct mode."""
        summary_hits = self.retrieve_summaries(question, where=summary_where)
        txn_hits = (
            self.retrieve_transactions(question, where=txn_where)
            if include_transactions else []
        )
        context = _build_context(summary_hits, txn_hits)

        messages: list[dict] = [{"role": "system", "content": _SYSTEM_DIRECT}]
        if use_memory and self.history:
            messages.extend(self.history[-6:])
        messages.append({
            "role":    "user",
            "content": f"Context (from Superstore sales database):\n{context}\n\nQuestion: {question}",
        })
        return messages, summary_hits, txn_hits

    def _run_direct(
        self,
        question:             str,
        summary_where:        dict | None,
        txn_where:            dict | None,
        include_transactions: bool,
        use_memory:           bool,
    ) -> tuple[str, list[dict], list[dict]]:
        messages, summary_hits, txn_hits = self._prepare_direct(
            question, summary_where, txn_where, include_transactions, use_memory,
        )
        answer = self.llm.chat(messages).content or ""
        self._update_history(question, answer, use_memory)
        return answer, summary_hits, txn_hits

    # ------------------------------------------------------------------
    # Internal — agent mode
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        question:   str,
        use_memory: bool,
    ) -> tuple[str, list[dict], list[dict]]:
        tool_schemas  = [t.to_ollama_schema() for t in self.tools.values()]
        summary_hits: list[dict] = []
        txn_hits:     list[dict] = []

        messages: list = [{"role": "system", "content": _SYSTEM_AGENT}]
        if use_memory and self.history:
            messages.extend(self.history[-6:])
        messages.append({"role": "user", "content": question})

        while True:
            response = self.llm.chat(messages, tools=tool_schemas)

            if not response.tool_calls:
                answer = response.content or ""
                self._update_history(question, answer, use_memory)
                return answer, summary_hits, txn_hits

            messages.append(response.raw_message)

            for tc in response.tool_calls:
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

                messages.append(self.llm.make_tool_message(tc, result_text))

    # ------------------------------------------------------------------
    # Conversation memory
    # ------------------------------------------------------------------

    def _update_history(self, question: str, answer: str, use_memory: bool) -> None:
        if use_memory:
            self.history.append({"role": "user",      "content": question})
            self.history.append({"role": "assistant",  "content": answer})
