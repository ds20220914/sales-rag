"""
Streamlit chat interface for the Superstore RAG pipeline.

Usage:
    streamlit run src/rag/app.py

What the UI controls:  provider (ollama / openai), model, retrieval mode.
What stays in .env:    API key, base URL, Ollama host — never exposed in UI.
"""

import os
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.dirname(_HERE)
_ROOT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

import streamlit as st
from rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Config from .env  (credentials — never shown in UI)
# ---------------------------------------------------------------------------

_DB_DIR         = os.environ.get("CHROMA_DB_PATH",
                                 os.path.join(_SRC_DIR, "vector_db", "chroma_db"))
_DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER",   "ollama")
_DEFAULT_OLLAMA   = os.environ.get("OLLAMA_MODEL",   "llama3.2:3b")
_DEFAULT_OPENAI   = os.environ.get("OPENAI_MODEL",   "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Model list helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def _ollama_models() -> list[str]:
    """Fetch locally pulled Ollama models (refreshed every 60 s)."""
    try:
        import ollama
        names = sorted(m.model for m in ollama.list().models)
        return names if names else [_DEFAULT_OLLAMA]
    except (ImportError, OSError, ConnectionRefusedError):
        return [_DEFAULT_OLLAMA]


def _openai_models() -> list[str]:
    """Return common OpenAI-compatible model names, env default first."""
    base = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    if _DEFAULT_OPENAI not in base:
        base.insert(0, _DEFAULT_OPENAI)
    return base


def _model_options(provider: str) -> list[str]:
    return _ollama_models() if provider == "ollama" else _openai_models()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_sources(summary_hits: list[dict], txn_hits: list[dict]) -> None:
    total = len(summary_hits) + len(txn_hits)
    with st.expander(f"Sources ({total} retrieved)", expanded=False):
        if summary_hits:
            st.markdown("**Aggregated summaries**")
            for src in summary_hits:
                label = "HIGH" if src["distance"] < 0.5 else "MED" if src["distance"] < 0.9 else "LOW"
                st.markdown(f"[{label}] `{src['id']}` dist={src['distance']:.4f}")
                st.caption(src["text"][:300])
        if txn_hits:
            st.markdown("**Transaction-level examples**")
            for src in txn_hits:
                label = "HIGH" if src["distance"] < 0.5 else "MED" if src["distance"] < 0.9 else "LOW"
                st.markdown(f"[{label}] `{src['id']}` dist={src['distance']:.4f}")
                st.caption(src["text"][:300])


# ---------------------------------------------------------------------------
# Pipeline cache  (keyed by provider + model + mode; credentials from .env)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_pipeline(provider: str, model: str, mode: str) -> RAGPipeline:
    return RAGPipeline(persist_dir=_DB_DIR, provider=provider,
                       model=model, mode=mode)


def _get_rag() -> RAGPipeline:
    s = st.session_state
    return _load_pipeline(s.provider, s.model, s.mode)


def _reset_pipeline() -> None:
    _load_pipeline.clear()
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "messages": [],
        "provider": _DEFAULT_PROVIDER,
        "model":    _DEFAULT_OLLAMA if _DEFAULT_PROVIDER == "ollama" else _DEFAULT_OPENAI,
        "mode":     "direct",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# on_change callbacks  (run before the next rerun, so state is already
# correct when widgets further down the sidebar are rendered)
# ---------------------------------------------------------------------------

def _on_provider_change() -> None:
    new_provider = st.session_state._w_provider
    st.session_state.provider = new_provider
    st.session_state.model = (
        _DEFAULT_OLLAMA if new_provider == "ollama" else _DEFAULT_OPENAI
    )
    _reset_pipeline()


def _on_model_change() -> None:
    st.session_state.model = st.session_state._w_model
    _reset_pipeline()


def _on_mode_change() -> None:
    st.session_state.mode = st.session_state._w_mode
    _reset_pipeline()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[dict | None, bool]:
    s = st.session_state
    with st.sidebar:
        st.title("Superstore RAG")
        st.divider()

        # --- Provider ---
        st.subheader("Provider")
        st.radio(
            "Provider", ["ollama", "openai"],
            index=0 if s.provider == "ollama" else 1,
            key="_w_provider",
            on_change=_on_provider_change,
            label_visibility="collapsed",
        )

        # --- Model ---
        st.subheader("Model")
        options = _model_options(s.provider)
        current = s.model if s.model in options else options[0]
        st.selectbox(
            "Model", options,
            index=options.index(current),
            key="_w_model",
            on_change=_on_model_change,
            label_visibility="collapsed",
        )

        st.divider()

        # --- Retrieval mode ---
        st.subheader("Retrieval mode")
        st.radio(
            "Mode", ["direct", "agent"],
            index=0 if s.mode == "direct" else 1,
            key="_w_mode",
            on_change=_on_mode_change,
            help=(
                "**direct** — pipeline retrieves context then calls LLM once.\n\n"
                "**agent** — LLM decides which tools to call and with what filters."
            ),
        )
        mode = s.mode

        st.divider()

        # --- Metadata filter (direct mode only) ---
        where: dict | None = None
        if mode == "direct":
            st.subheader("Metadata Filter (optional)")
            filter_type = st.selectbox("Filter by", ["None", "Category", "Region", "Year"])
            if filter_type == "Category":
                where = {"category": st.selectbox("Category",
                         ["Technology", "Furniture", "Office Supplies"])}
            elif filter_type == "Region":
                where = {"region": st.selectbox("Region",
                         ["West", "East", "Central", "South"])}
            elif filter_type == "Year":
                where = {"year": st.selectbox("Year",
                         ["2014", "2015", "2016", "2017"])}
        else:
            st.caption("In agent mode the LLM decides its own filters.")

        st.divider()

        show_sources = st.toggle("Show retrieved sources", value=True)

        if st.button("Clear conversation"):
            _get_rag().reset_memory()
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Example questions")
        for ex in [
            "What is the sales trend from 2014 to 2017?",
            "Which season has the highest sales?",
            "Which category generates the most revenue?",
            "Which sub-categories have the highest profit margins?",
            "Which region performs best in terms of profit?",
            "Compare Technology and Furniture sales trends.",
            "How do West and East regions compare in profit?",
        ]:
            if st.button(ex, use_container_width=True):
                st.session_state.prefill = ex
                st.rerun()

    return where, show_sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Superstore Sales RAG", page_icon="📊", layout="wide")
    _init_state()

    where, show_sources = _sidebar()
    s    = st.session_state
    mode = s.mode

    st.title("📊 Superstore Sales Analysis")
    st.caption(f"Provider: `{s.provider}` · Model: `{s.model}` · Mode: `{mode}`")

    for msg in s.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_sources and (msg.get("summary_hits") or msg.get("txn_hits")):
                _render_sources(msg.get("summary_hits", []), msg.get("txn_hits", []))

    question = st.chat_input("Ask a question about Superstore sales...") \
               or s.pop("prefill", None)
    if not question:
        return

    s.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    rag = _get_rag()
    with st.chat_message("assistant"):
        if mode == "agent":
            with st.spinner("Agent thinking and retrieving..."):
                answer = st.write_stream(rag.stream(question, use_memory=True))
        else:
            answer = st.write_stream(
                rag.stream(question, summary_where=where, use_memory=True)
            )
        if show_sources:
            _render_sources(rag.last_summary_hits, rag.last_txn_hits)

    s.messages.append({
        "role":         "assistant",
        "content":      answer,
        "summary_hits": rag.last_summary_hits,
        "txn_hits":     rag.last_txn_hits,
    })


if __name__ == "__main__":
    main()
