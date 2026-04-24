# RAG Pipeline

This document covers the modules under `src/rag/` and `src/llm/` that implement the retrieval-augmented generation pipeline.

```
src/
├── llm/
│   ├── base.py        # Abstract base class + shared data structures
│   ├── ollama.py      # Ollama provider (local models)
│   ├── openai.py      # OpenAI-compatible provider
│   └── __init__.py    # make_llm() factory, public API
└── rag/
    ├── tools.py       # RetrievalTool definitions and JSON schemas
    ├── pipeline.py    # RAGPipeline: direct mode, agent mode, streaming
    └── app.py         # Streamlit chat interface
```

---

## LLM abstraction (`src/llm/`)

The `llm` package decouples the pipeline from any specific model SDK. Adding a new provider requires only a new file that subclasses `LLMProvider`.

### Data structures (`base.py`)

#### `ToolCall`

```python
@dataclass
class ToolCall:
    id:        str   # call ID (required by OpenAI for tool-result routing)
    name:      str   # tool function name
    arguments: dict  # parsed argument dictionary
```

#### `ChatResponse`

```python
@dataclass
class ChatResponse:
    content:     str | None      # text response (None when tool calls are present)
    tool_calls:  list[ToolCall]  # empty list if the model returned text only
    raw_message: Any             # provider-specific message object for history appending
```

`raw_message` is the object that must be appended to the messages list so the model can see its own tool-call decisions in subsequent turns. For Ollama this is the SDK's `Message` object; for OpenAI it is a serialisable dict.

---

### `LLMProvider` abstract base (`base.py`)

Every concrete provider must implement three methods:

| Method | Signature | Purpose |
|---|---|---|
| `chat` | `(messages, tools=None) → ChatResponse` | Single-shot completion; supports function/tool calls |
| `stream_chat` | `(messages) → Generator[str, None, None]` | Streaming completion; yields text tokens; tool calls not supported |
| `make_tool_message` | `(tool_call, content) → dict` | Formats a tool-result message for the next turn |

The `make_tool_message` signature differs by provider because OpenAI requires `tool_call_id` in the result message while Ollama does not.

---

### `OllamaProvider` (`ollama.py`)

Wraps the `ollama` Python SDK for local model inference.

**Constructor parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` | `"llama3.2:3b"` | Ollama model tag |
| `host` | `None` | Ollama server URL; `None` uses the SDK default (`http://localhost:11434`) |

**`chat()`** calls `ollama.chat()` with the message list and optional tool schemas. Ollama does not expose tool-call IDs, so a sequential integer index is used as a surrogate `id` in `ToolCall`.

**`stream_chat()`** calls `ollama.chat(..., stream=True)` and yields `chunk.message.content` for each non-empty chunk.

**`make_tool_message()`** returns `{"role": "tool", "content": result}` — no `tool_call_id` required.

---

### `OpenAIProvider` (`openai.py`)

Wraps the `openai` Python SDK. Compatible with any OpenAI-format endpoint.

**Constructor parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` | `"gpt-4o-mini"` | Model name as expected by the endpoint |
| `api_key` | `None` | API key; `None` falls back to the `OPENAI_API_KEY` environment variable |
| `base_url` | `None` | Custom endpoint base URL; `None` uses `https://api.openai.com/v1` |

**Compatible endpoints:**

| Service | `base_url` |
|---|---|
| OpenAI | *(omit)* |
| Groq | `https://api.groq.com/openai/v1` |
| Together AI | `https://api.together.xyz/v1` |
| LM Studio | `http://localhost:1234/v1` |
| Ollama REST | `http://localhost:11434/v1` |

**`chat()`** calls `client.chat.completions.create()`. The tool arguments returned by the API are a JSON string; the provider parses them to a dict before returning. A serialisable `raw_message` dict is constructed because the SDK response object is not JSON-serialisable. When the response contains tool calls, `content` is set to `""` (empty string) rather than `None` for compatibility with strict API proxies.

**`stream_chat()`** calls `completions.create(..., stream=True)` and yields `chunk.choices[0].delta.content` for each chunk, skipping chunks where `choices` is empty (which occurs at the end-of-stream signal from some compatible APIs).

**`make_tool_message()`** returns `{"role": "tool", "tool_call_id": tc.id, "content": result}`.

---

### `make_llm()` factory (`__init__.py`)

```python
from llm import make_llm

llm = make_llm("ollama", model="llama3.2:3b")
llm = make_llm("openai", model="gpt-4o-mini", api_key="sk-...")
llm = make_llm("openai", model="llama3-8b-8192",
               api_key="gsk_...", base_url="https://api.groq.com/openai/v1")
```

Dispatches to the correct provider class via a `_PROVIDERS` registry dict. Raises `ValueError` for unknown provider names. Keyword arguments are forwarded directly to the provider constructor.

---

## Retrieval tools (`src/rag/tools.py`)

### `RetrievalTool` dataclass

Bundles a JSON Schema, a human-readable description, and a callable into a single object that can be used both as a Python callable (direct mode) and as a function-call schema handed to the LLM (agent mode).

```python
@dataclass
class RetrievalTool:
    name:        str
    description: str
    parameters:  dict      # JSON Schema for the function arguments
    _fn:         Callable  # underlying ChromaDB query function
```

**`__call__(query, where=None, n_results=None)`** — delegates to `_fn` with the provided arguments.

**`to_ollama_schema()`** — returns the OpenAI-compatible function-call dict:

```python
{
    "type": "function",
    "function": {
        "name":        self.name,
        "description": self.description,
        "parameters":  self.parameters,
    },
}
```

This format is accepted by both Ollama and OpenAI-compatible APIs.

---

### Available tools

#### `search_summaries`

Searches the `summaries` collection for aggregated analytical documents.

- **Use for:** trends, rankings, category/region comparisons, profit analysis, seasonal patterns
- **Default top-k:** 5
- **Valid `where` keys:** `type`, `year`, `month`, `quarter`, `season`, `region`, `category`, `sub_category`, `segment`, `state`

#### `search_transactions`

Searches the `transactions` collection for individual order-level documents.

- **Use for:** concrete examples, specific customer/product details, discount verification
- **Default top-k:** 3
- **Valid `where` keys:** `year`, `month`, `region`, `category`, `sub_category`, `segment`, `state`

### `make_tools(col_summaries, col_transactions) → dict[str, RetrievalTool]`

Binds the two tools to live ChromaDB collection objects and returns them as a dict keyed by tool name. Called once during `RAGPipeline.__init__` so the collections remain open and reused across all queries.

---

## RAG Pipeline (`src/rag/pipeline.py`)

### Configuration

`RAGPipeline` reads its defaults from `.env` at import time:

| `.env` key | Pipeline parameter | Fallback |
|---|---|---|
| `CHROMA_DB_PATH` | `persist_dir` | `src/vector_db/chroma_db` |
| `LLM_PROVIDER` | `provider` | `"ollama"` |
| `OLLAMA_MODEL` | `model` (when provider=ollama) | `"llama3.2:3b"` |
| `OPENAI_MODEL` | `model` (when provider=openai) | `"gpt-4o-mini"` |
| `OPENAI_API_KEY` | injected into `OpenAIProvider` | — |
| `OPENAI_BASE_URL` | injected into `OpenAIProvider` | — |
| `OLLAMA_HOST` | injected into `OllamaProvider` | — |

Credentials (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OLLAMA_HOST`) are injected automatically inside `__init__` via `setdefault`, so callers never need to pass them explicitly.

### Constructor

```python
RAGPipeline(
    persist_dir = ...,   # ChromaDB directory
    provider    = ...,   # 'ollama' or 'openai'
    model       = ...,   # model name
    mode        = "direct",  # 'direct' or 'agent'
    n_summary   = 5,     # top-k for search_summaries
    n_txn       = 2,     # top-k for search_transactions
    **llm_kwargs,        # forwarded to make_llm()
)
```

On construction, the pipeline:
1. Instantiates the LLM provider via `make_llm()`
2. Opens the ChromaDB client and both collections
3. Binds retrieval tools via `make_tools()`
4. Initialises conversation history and last-hit caches

---

### Operating modes

#### Direct mode

```
User question
    │
    ├──▶ retrieve_summaries(question, where=summary_where)   top-5
    ├──▶ retrieve_transactions(question, where=txn_where)    top-2
    │
    ▼
_build_context()  →  two-section markdown block
    │
    ▼
System prompt + history (last 6 turns) + context + question
    │
    ▼
llm.chat() / llm.stream_chat()
    │
    ▼
Answer text
```

Both collections are always queried before calling the LLM. The combined context is injected into a single user message alongside the question. This mode is **predictable and fast** — exactly one LLM call per question.

#### Agent mode

```
User question
    │
    ▼
System prompt + history + question  ──▶  llm.chat(tools=[...])
                                              │
                                    ┌─────────┴──────────┐
                                    │ tool calls present? │
                                    └─────────┬──────────┘
                               yes ◀──────────┤
                                    │         │ no → return answer
                                    ▼
                             execute tools
                             append results
                                    │
                                    └──▶ llm.chat() again (loop)
                                              (max 6 iterations)
```

The LLM receives tool schemas and decides autonomously which tools to call, with what queries, and with what metadata filters. The loop continues until the model stops calling tools (returns plain text) or the 6-iteration guard is reached. If the guard triggers, the model is prompted to summarise its findings with what it already has.

**Why 6 iterations?** A typical analytical question requires 1–3 tool calls. 6 provides headroom for comparative questions that need separate calls per dimension (e.g. West vs East requires two separate `search_summaries` calls) while preventing runaway loops on poorly-supported models.

---

### Public API

#### `ask(question, summary_where=None, txn_where=None, include_transactions=True, use_memory=True) → dict`

Single-shot, blocking. Works in both direct and agent mode.

Returns:
```python
{
    "answer":       str,        # LLM response text
    "summary_hits": list[dict], # retrieved summary documents
    "txn_hits":     list[dict], # retrieved transaction documents
    "mode":         str,        # 'direct' or 'agent'
}
```

After returning, `last_summary_hits` and `last_txn_hits` properties are also updated.

---

#### `stream(question, summary_where=None, txn_where=None, include_transactions=True, use_memory=True) → Generator[str, None, None]`

Streaming version. Yields text tokens as they arrive from the LLM.

- **Direct mode:** true token-by-token streaming via `llm.stream_chat()`
- **Agent mode:** not streamable (tool calls must complete before a response can be formed); falls back to `ask()` and yields the full answer as a single chunk

After the generator is exhausted, `last_summary_hits` and `last_txn_hits` are populated.

---

#### `stream_agent(question, use_memory=True) → Generator[dict, None, None]`

Agent-mode only. Yields structured events for real-time UI rendering instead of raw text.

Event types:

| Event | Fields | Timing |
|---|---|---|
| `tool_call` | `name`, `query`, `where` | Immediately when the LLM decides to call a tool |
| `tool_result` | `name`, `n_hits`, `preview` | After the tool executes |
| `answer` | `text` | When the final answer is ready |

The Streamlit app uses this method to populate a live `st.status` block showing each tool invocation and result before the final answer appears.

---

#### `retrieve_summaries(query, where=None, n_results=None) → list[dict]`
#### `retrieve_transactions(query, where=None, n_results=None) → list[dict]`

Direct access to the retrieval tools. Useful for testing or for scripts that need raw retrieval without LLM generation.

---

#### `reset_memory() → None`

Clears the conversation history. Called when the user clicks "Clear conversation" in the UI.

---

### Conversation memory

Up to the **last 6 message turns** (3 user + 3 assistant) are prepended to the prompt on each call when `use_memory=True`. The history is stored as a plain list of `{"role": ..., "content": ...}` dicts on the pipeline instance.

Memory is intentionally shallow (6 turns) to avoid token-limit issues with smaller local models.

---

### Internal helpers

#### `_prepare_direct()`

Shared by both `_run_direct()` and `stream()`. Retrieves hits, builds the context string, and assembles the final message list. Extracting this prevents the identical message-building logic from being duplicated.

#### `_parse_where(where)`

Normalises the `where` argument received from an LLM tool call. Two defences:

1. **String deserialisation:** The LLM sometimes encodes the filter as a JSON string rather than a dict; this function parses it if so.
2. **Contradictory `$and` detection:** A filter like `{"$and": [{"region": "West"}, {"region": "East"}]}` matches the same key against two different values, which is always empty. The function detects this and returns `None` (no filter) so the tool returns unfiltered results instead of zero results.

---

## System prompts

### Direct mode

> You are a business intelligence analyst specialized in retail sales data. You answer questions about the Superstore dataset (2014–2017, ~10,000 transactions). Base your answers strictly on the provided context snippets. Be concise, cite specific numbers, and clearly label years, categories, or regions. If the context lacks sufficient data to answer, say so explicitly.

### Agent mode

> You are a business intelligence analyst with access to a Superstore sales database (2014–2017). Use the available tools to retrieve the data you need before answering. Guidelines: Call search_summaries for aggregate statistics, trends, rankings, and comparisons. Call search_transactions for concrete order examples or to verify individual-level claims. You may call tools multiple times with different queries or filters. Once you have sufficient data, give a concise, number-backed answer.

---

## End-to-end flow

```
User question (from Streamlit or script)
         │
         ▼
RAGPipeline.ask() / .stream() / .stream_agent()
         │
         ├── direct ──▶ _prepare_direct()
         │                   ├── retrieve_summaries()  ──▶ store.query("summaries")
         │                   └── retrieve_transactions() ─▶ store.query("transactions")
         │              llm.chat() or llm.stream_chat()
         │
         └── agent ───▶ _run_agent() / stream_agent()
                             └── loop (max 6 iters):
                                   llm.chat(tools=[...])
                                   ├── tool_call → execute → append result
                                   └── no tool_call → return answer
         │
         ▼
Answer text  +  last_summary_hits  +  last_txn_hits
```
