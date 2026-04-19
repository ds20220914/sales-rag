"""
Embedding model comparison for the Sales RAG project.

Builds a ChromaDB summaries+transactions index for each candidate model,
runs all retrieval test cases, collects metrics, and writes a Markdown
report to docs/embedding_comparison.md.

Usage:
    python src/vector_db/compare_embeddings.py [--persist-dir ./chroma_eval]

Models tested:
    all-MiniLM-L6-v2          baseline  22 M params  384-dim
    multi-qa-MiniLM-L6-cos-v1 QA-tuned  22 M params  384-dim
    all-mpnet-base-v2          quality  110 M params  768-dim
    BAAI/bge-small-en-v1.5     SOTA      33 M params  384-dim
"""

import argparse
import os
import sys
import time
import shutil
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
_DEFAULT_EVAL_DIR = os.path.join(_HERE, "chroma_eval")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_processing.loader import load_data
from data_processing.text_converter import build_all_texts
from data_processing.chunker import chunk_documents
from vector_db.store import upsert_chunks
from vector_db.retrieval_test import TEST_CASES

# ---------------------------------------------------------------------------
# Models to evaluate
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "label": "MiniLM-L6 (baseline)",
        "params": "22 M",
        "dims": 384,
    },
    {
        "name": "multi-qa-MiniLM-L6-cos-v1",
        "label": "multi-qa-MiniLM-L6 (QA-tuned)",
        "params": "22 M",
        "dims": 384,
    },
    {
        "name": "all-mpnet-base-v2",
        "label": "mpnet-base-v2 (high-quality)",
        "params": "110 M",
        "dims": 768,
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "label": "bge-small-en-v1.5 (SOTA small)",
        "params": "33 M",
        "dims": 384,
    },
]

RELEVANCE = lambda d: "HIGH" if d < 0.5 else ("MED" if d < 0.9 else "LOW")


# ---------------------------------------------------------------------------
# Per-model result
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    label: str
    collection: str
    query: str
    top1_id: str
    top1_dist: float
    top1_relevance: str


@dataclass
class ModelResult:
    model_name: str
    model_label: str
    params: str
    dims: int
    index_time_s: float
    query_results: list[QueryResult] = field(default_factory=list)

    @property
    def avg_top1_dist(self) -> float:
        return sum(r.top1_dist for r in self.query_results) / len(self.query_results)

    @property
    def high_count(self) -> int:
        return sum(1 for r in self.query_results if r.top1_relevance == "HIGH")

    @property
    def med_count(self) -> int:
        return sum(1 for r in self.query_results if r.top1_relevance == "MED")

    @property
    def low_count(self) -> int:
        return sum(1 for r in self.query_results if r.top1_relevance == "LOW")


# ---------------------------------------------------------------------------
# Build index for one model
# ---------------------------------------------------------------------------
def build_model_index(
    model_name: str,
    persist_dir: str,
    transaction_chunks,
    summary_chunks,
) -> float:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    client = chromadb.PersistentClient(path=persist_dir)

    t0 = time.time()
    for col_name, chunks in [("transactions", transaction_chunks),
                              ("summaries",    summary_chunks)]:
        col = client.get_or_create_collection(
            name=col_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        upsert_chunks(col, chunks)

    return time.time() - t0


# ---------------------------------------------------------------------------
# Run all test cases for one model
# ---------------------------------------------------------------------------
def run_model_queries(model_name: str, persist_dir: str) -> list[QueryResult]:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    client = chromadb.PersistentClient(path=persist_dir)
    collections = {
        name: client.get_or_create_collection(
            name=name, embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        for name in ("transactions", "summaries")
    }

    results = []
    for label, col_name, q, where in TEST_CASES:
        col = collections[col_name]
        kwargs = {"query_texts": [q], "n_results": 1}
        if where:
            kwargs["where"] = where
        raw = col.query(**kwargs)

        if raw["ids"][0]:
            dist = raw["distances"][0][0]
            doc_id = raw["ids"][0][0]
        else:
            dist = 1.0
            doc_id = "(no result)"

        results.append(QueryResult(
            label=label,
            collection=col_name,
            query=q,
            top1_id=doc_id,
            top1_dist=dist,
            top1_relevance=RELEVANCE(dist),
        ))
    return results


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------
def write_report(model_results: list[ModelResult], out_path: str) -> None:
    lines = []
    a = lines.append

    a("# Embedding Model Comparison — Sales RAG\n")
    a(f"_Generated automatically by `compare_embeddings.py`_\n")

    # Overview table
    a("## Model Overview\n")
    a("| Model | Params | Dims | Index time (s) | HIGH | MED | LOW | Avg top-1 dist |")
    a("|-------|--------|------|---------------|------|-----|-----|----------------|")
    for mr in model_results:
        a(f"| {mr.model_label} | {mr.params} | {mr.dims} "
          f"| {mr.index_time_s:.0f} "
          f"| {mr.high_count} | {mr.med_count} | {mr.low_count} "
          f"| {mr.avg_top1_dist:.4f} |")
    a("")

    # Per-query comparison table
    a("## Per-Query Top-1 Distance\n")
    header = "| # | Query | " + " | ".join(
        mr.model_label.split(" ")[0] for mr in model_results
    ) + " |"
    sep = "|---|-------" + "|-------" * len(model_results) + "|"
    a(header)
    a(sep)

    n = len(TEST_CASES)
    for i in range(n):
        label = model_results[0].query_results[i].label
        cells = []
        for mr in model_results:
            r = mr.query_results[i]
            cells.append(f"{r.top1_dist:.4f} [{r.top1_relevance}]")
        a(f"| {i+1:02d} | {label} | " + " | ".join(cells) + " |")
    a("")

    # Per-query top-1 document comparison
    a("## Per-Query Top-1 Document Retrieved\n")
    header = "| # | Query | " + " | ".join(
        mr.model_label.split(" ")[0] for mr in model_results
    ) + " |"
    a(header)
    a(sep)

    for i in range(n):
        label = model_results[0].query_results[i].label
        cells = [mr.query_results[i].top1_id for mr in model_results]
        a(f"| {i+1:02d} | {label} | " + " | ".join(cells) + " |")
    a("")

    # Analysis section
    a("## Analysis\n")

    # Best model by avg distance
    best = min(model_results, key=lambda m: m.avg_top1_dist)
    a(f"**Best average top-1 distance**: {best.model_label} ({best.avg_top1_dist:.4f})\n")

    # Fastest indexing
    fastest = min(model_results, key=lambda m: m.index_time_s)
    a(f"**Fastest indexing**: {fastest.model_label} ({fastest.index_time_s:.0f}s)\n")

    # Queries where models disagree on top-1 doc
    a("### Queries where top-1 document differs across models\n")
    a("| # | Query | Documents retrieved |")
    a("|---|-------|---------------------|")
    for i in range(n):
        docs = [mr.query_results[i].top1_id for mr in model_results]
        if len(set(docs)) > 1:
            label = model_results[0].query_results[i].label
            doc_str = " / ".join(
                f"{mr.model_label.split(' ')[0]}: `{mr.query_results[i].top1_id}`"
                for mr in model_results
            )
            a(f"| {i+1:02d} | {label} | {doc_str} |")
    a("")

    # Queries where a model underperforms (MED or LOW when others are HIGH)
    a("### Queries where at least one model returned MED/LOW\n")
    a("| # | Query | " + " | ".join(
        mr.model_label.split(" ")[0] for mr in model_results
    ) + " |")
    a(sep)
    found = False
    for i in range(n):
        relevances = [mr.query_results[i].top1_relevance for mr in model_results]
        if any(r != "HIGH" for r in relevances):
            found = True
            label = model_results[0].query_results[i].label
            a(f"| {i+1:02d} | {label} | " + " | ".join(relevances) + " |")
    if not found:
        a("_All models returned HIGH relevance on every query._")
    a("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(persist_root: str) -> None:
    print("Loading data and building chunks (once)...")
    df = load_data()
    transactions, summaries = build_all_texts(df)
    tx_chunks = chunk_documents(transactions, chunk_size=1000)
    su_chunks = chunk_documents(summaries,    chunk_size=1000)
    print(f"  {len(tx_chunks):,} transaction chunks, {len(su_chunks):,} summary chunks\n")

    all_results: list[ModelResult] = []

    for m in MODELS:
        name = m["name"]
        safe_name = name.replace("/", "_").replace("-", "_")
        persist_dir = os.path.join(persist_root, safe_name)

        print(f"{'='*60}")
        print(f"Model : {name}")
        print(f"Dir   : {persist_dir}")

        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        print("  Building index...", end=" ", flush=True)
        elapsed = build_model_index(name, persist_dir, tx_chunks, su_chunks)
        print(f"{elapsed:.1f}s")

        print("  Running queries...", end=" ", flush=True)
        qr = run_model_queries(name, persist_dir)
        high = sum(1 for r in qr if r.top1_relevance == "HIGH")
        print(f"done  ({high}/{len(qr)} HIGH)")

        all_results.append(ModelResult(
            model_name=name,
            model_label=m["label"],
            params=m["params"],
            dims=m["dims"],
            index_time_s=elapsed,
            query_results=qr,
        ))

    report_path = os.path.join(
        os.path.dirname(_SRC_DIR), "docs", "embedding_comparison.md"
    )
    write_report(all_results, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", default=_DEFAULT_EVAL_DIR)
    args = parser.parse_args()
    main(args.persist_dir)