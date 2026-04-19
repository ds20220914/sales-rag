"""
ChromaDB vector store wrapper.

Two collections are maintained:
  - "transactions" : one document per sales row
  - "summaries"    : all aggregated summary documents

Embedding model: BAAI/bge-small-en-v1.5
  33M params, 384-dim, cosine distance.
  Chosen over all-MiniLM-L6-v2 (baseline) based on embedding comparison:
  avg top-1 distance 0.1796 vs 0.2701 across 13 retrieval test cases.
"""

import chromadb
from chromadb.utils import embedding_functions

_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_COLLECTIONS = ("transactions", "summaries")


def get_client(persist_dir: str = "./chroma_db") -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_dir)


def get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)


def get_collection(
    client: chromadb.PersistentClient,
    name: str,
    ef: embedding_functions.SentenceTransformerEmbeddingFunction,
) -> chromadb.Collection:
    if name not in _COLLECTIONS:
        raise ValueError(f"Unknown collection '{name}'. Choose from {_COLLECTIONS}.")
    return client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(collection: chromadb.Collection, chunks: list) -> None:
    """Write Chunk objects to a collection in batches (ChromaDB max ~5000/call)."""
    BATCH = 2000
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )


def query(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Similarity search with optional metadata filter.

    `where` uses ChromaDB filter syntax, e.g.:
        {"region": "West"}
        {"$and": [{"category": "Technology"}, {"year": "2017"}]}

    Returns a list of dicts with keys: id, text, metadata, distance.
    """
    kwargs = {"query_texts": [query_text], "n_results": n_results}
    if where:
        kwargs["where"] = where

    result = collection.query(**kwargs)

    docs = []
    for doc_id, text, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        docs.append({"id": doc_id, "text": text, "metadata": meta, "distance": dist})
    return docs