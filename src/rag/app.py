"""
Streamlit chat interface for the Superstore RAG pipeline.

Usage:
    streamlit run src/rag/app.py

Optional env vars:
    CHROMA_DB_DIR  : Path to ChromaDB dir (default: src/vector_db/chroma_db)

    All other configuration (API key, model, Ollama host) is entered via the UI.
    Model defaults live in the adapters:
        llm/openai_adapter.py  — default model: gpt-4o-mini
        llm/ollama_adapter.py  — OLLAMA_HOST (default: http://localhost:11434)
                               — default model: llama3.2:3b

Architecture
------------
app.py      — Streamlit UI; selects provider (OpenAI / Ollama) and mode
pipeline.py — RAG orchestration; calls the injected LLM adapter
llm/        — provider adapters (openai_adapter.py, ollama_adapter.py)
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import streamlit as st

from llm import make_adapter
from rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB = os.path.join(_SRC_DIR, "vector_db", "chroma_db")
_DB_DIR     = os.environ.get("CHROMA_DB_DIR", _DEFAULT_DB)

# UI model lists — display only, actual defaults live in the adapters
_OPENAI_MODELS = ["gpt-4o-mini"]
_OLLAMA_MODELS = ["llama3.2:3b"]


# ---------------------------------------------------------------------------
# Source rendering
# ---------------------------------------------------------------------------

def _render_sources(summary_hits: list[dict], txn_hits: list[dict]) -> None:
    total = len(summary_hits) + len(txn_hits)
    with st.expander(f"Sources ({total} retrieved)", expanded=False):
        if summary_hits:
            st.markdown("**Aggregated summaries**")
            for src in summary_hits:
                dist_label = "HIGH" if src["distance"] < 0.5 else "MED" if src["distance"] < 0.9 else "LOW"
                st.markdown(f"[{dist_label}] `{src['id']}` dist={src['distance']:.4f}")
                st.caption(src["text"][:300])
        if txn_hits:
            st.markdown("**Transaction-level examples**")
            for src in txn_hits:
                dist_label = "HIGH" if src["distance"] < 0.5 else "MED" if src["distance"] < 0.9 else "LOW"
                st.markdown(f"[{dist_label}] `{src['id']}` dist={src['distance']:.4f}")
                st.caption(src["text"][:300])


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "messages":     [],
        "mode":         "direct",
        "provider":     "openai",
        "openai_api_key": "",
        "openai_model": _OPENAI_MODELS[0],
        "ollama_model": _OLLAMA_MODELS[0],
        "ollama_host":  "http://localhost:11434",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def _load_pipeline(
    provider: str,
    model: str,
    mode: str,
    api_key: str,
    ollama_host: str,
) -> RAGPipeline:
    """
    Cache one pipeline per unique (provider, model, mode, api_key, ollama_host).
    Changing any of these creates a fresh pipeline automatically.
    """
    adapter = make_adapter(
        provider=provider,
        model=model,
        api_key=api_key or None,
        ollama_host=ollama_host or None,
    )
    return RAGPipeline(llm=adapter, persist_dir=_DB_DIR, mode=mode)


def _reset_chat() -> None:
    """Clear conversation history from session state and pipeline memory."""
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[dict | None, bool, str]:
    with st.sidebar:
        st.title("Superstore RAG")
        st.divider()

        # ----------------------------------------------------------------
        # Provider selection
        # ----------------------------------------------------------------
        st.subheader("🧠 LLM Provider")
        provider = st.radio(
            "Provider",
            ["openai", "ollama"],
            index=0 if st.session_state.provider == "openai" else 1,
            format_func=lambda p: "☁️  OpenAI" if p == "openai" else "🏠  Ollama (local)",
            help="OpenAI uses cloud models. Ollama runs models locally on your machine.",
        )
        if provider != st.session_state.provider:
            st.session_state.provider = provider
            _reset_chat()
            st.rerun()

        # ----------------------------------------------------------------
        # Provider-specific settings
        # ----------------------------------------------------------------
        if provider == "openai":
            st.subheader("🔑 OpenAI API Key")
            api_key_input = st.text_input(
                "API Key",
                value=st.session_state.openai_api_key,
                type="password",
                placeholder="sk-...",
                help="Get a key at https://platform.openai.com/api-keys",
            )
            if api_key_input != st.session_state.openai_api_key:
                st.session_state.openai_api_key = api_key_input
                _reset_chat()
                st.rerun()

            if not st.session_state.openai_api_key:
                st.warning(
                    "No API key set. Get one at "
                    "[platform.openai.com](https://platform.openai.com/api-keys).",
                    icon="⚠️",
                )

            st.subheader("🤖 OpenAI Model")
            selected_model = st.selectbox(
                "Model",
                _OPENAI_MODELS,
                index=_OPENAI_MODELS.index(st.session_state.openai_model)
                if st.session_state.openai_model in _OPENAI_MODELS else 0,
                help="gpt-4o-mini is the fastest and most cost-effective.",
            )
            if selected_model != st.session_state.openai_model:
                st.session_state.openai_model = selected_model
                _reset_chat()
                st.rerun()

        else:  # ollama
            st.subheader("🏠 Ollama Settings")
            ollama_host_input = st.text_input(
                "Ollama host",
                value=st.session_state.ollama_host,
                placeholder="http://localhost:11434",
            )
            if ollama_host_input != st.session_state.ollama_host:
                st.session_state.ollama_host = ollama_host_input
                _reset_chat()
                st.rerun()

            selected_ollama_model = st.selectbox(
                "Ollama model",
                _OLLAMA_MODELS,
                index=_OLLAMA_MODELS.index(st.session_state.ollama_model)
                if st.session_state.ollama_model in _OLLAMA_MODELS else 0,
                help="Model must already be pulled: `ollama pull <model>`",
            )
            if selected_ollama_model != st.session_state.ollama_model:
                st.session_state.ollama_model = selected_ollama_model
                _reset_chat()
                st.rerun()

            st.caption(
                "Make sure Ollama is running locally and the model is pulled. "
                "Tool-calling works on llama3.2, llama3.1, mistral, qwen2.5."
            )

        st.divider()

        # ----------------------------------------------------------------
        # Retrieval mode
        # ----------------------------------------------------------------
        st.subheader("Retrieval mode")
        mode = st.radio(
            "Mode",
            ["direct", "agent"],
            index=0 if st.session_state.mode == "direct" else 1,
            help=(
                "**direct** — pipeline calls both tools, builds context, LLM answers once.\n\n"
                "**agent** — LLM decides which tools to call and with what queries/filters. "
                "Requires a tool-capable model."
            ),
        )
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            _reset_chat()

        st.divider()

        # ----------------------------------------------------------------
        # Metadata filter (direct mode only)
        # ----------------------------------------------------------------
        where: dict | None = None
        if mode == "direct":
            st.subheader("Metadata Filter (optional)")
            filter_type = st.selectbox("Filter by", ["None", "Category", "Region", "Year"])
            if filter_type == "Category":
                cat = st.selectbox("Category", ["Technology", "Furniture", "Office Supplies"])
                where = {"category": cat}
            elif filter_type == "Region":
                reg = st.selectbox("Region", ["West", "East", "Central", "South"])
                where = {"region": reg}
            elif filter_type == "Year":
                yr = st.selectbox("Year", ["2014", "2015", "2016", "2017"])
                where = {"year": yr}
            st.divider()
        else:
            st.caption("In agent mode the LLM decides its own filters.")
            st.divider()

        show_sources = st.toggle("Show retrieved sources", value=True)

        if st.button("Clear conversation"):
            _reset_chat()
            # Also clear pipeline memory
            try:
                rag = _get_pipeline()
                rag.reset_memory()
            except Exception:
                pass
            st.rerun()

        st.divider()
        st.subheader("Example questions")
        examples = [
            "What is the sales trend from 2014 to 2017?",
            "Which season has the highest sales?",
            "Which category generates the most revenue?",
            "Which sub-categories have the highest profit margins?",
            "Which region performs best in terms of profit?",
            "Compare Technology and Furniture sales trends.",
            "How do West and East regions compare in profit?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["prefill"] = ex
                st.rerun()

    return where, show_sources, mode


# ---------------------------------------------------------------------------
# Pipeline accessor (reads from session state)
# ---------------------------------------------------------------------------

def _get_pipeline() -> RAGPipeline:
    """Build or retrieve the cached pipeline for the current session settings."""
    provider    = st.session_state.provider
    mode        = st.session_state.mode
    api_key     = st.session_state.openai_api_key if provider == "openai" else ""
    ollama_host = st.session_state.ollama_host    if provider == "ollama" else ""
    model = (
        st.session_state.openai_model
        if provider == "openai"
        else st.session_state.ollama_model
    )
    return _load_pipeline(
        provider=provider,
        model=model,
        mode=mode,
        api_key=api_key,
        ollama_host=ollama_host,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Superstore Sales RAG",
        page_icon="📊",
        layout="wide",
    )
    _init_state()
    where, show_sources, mode = _sidebar()

    provider = st.session_state.provider
    model    = (
        st.session_state.openai_model
        if provider == "openai"
        else st.session_state.ollama_model
    )

    st.title("📊 Superstore Sales Analysis")
    st.caption(
        f"Ask questions about the Superstore dataset (2014–2017) · "
        f"provider: `{provider}` · model: `{model}` · mode: `{mode}`"
    )

    # Guard: OpenAI needs an API key
    if provider == "openai" and not st.session_state.openai_api_key:
        st.info("👈 Enter your OpenAI API key in the sidebar.", icon="🔑")
        return

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_sources and (msg.get("summary_hits") or msg.get("txn_hits")):
                _render_sources(msg.get("summary_hits", []), msg.get("txn_hits", []))

    prefill  = st.session_state.pop("prefill", None)
    question = st.chat_input("Ask a question about Superstore sales...") or prefill

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    rag = _get_pipeline()

    with st.chat_message("assistant"):
        spinner_msg = (
            "Retrieving and generating..."
            if mode == "direct"
            else "Agent thinking and retrieving..."
        )
        with st.spinner(spinner_msg):
            result = rag.ask(question, summary_where=where, use_memory=True)
        st.markdown(result["answer"])
        if show_sources:
            _render_sources(result["summary_hits"], result["txn_hits"])

    st.session_state.messages.append({
        "role":         "assistant",
        "content":      result["answer"],
        "summary_hits": result["summary_hits"],
        "txn_hits":     result["txn_hits"],
    })


if __name__ == "__main__":
    main()