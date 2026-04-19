"""
One-shot script: build and persist the ChromaDB index.

Usage:
    python src/vector_db/build_index.py [--chunk-size 1000] [--persist-dir <path>]

Default persist directory: src/vector_db/chroma_db
"""

import argparse
import sys
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
_DEFAULT_DB = os.path.join(_HERE, "chroma_db")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_processing.loader import load_data
from data_processing.text_converter import build_all_texts
from data_processing.chunker import chunk_documents
from vector_db.store import get_client, get_embedding_function, get_collection, upsert_chunks, query


def build_index(chunk_size: int = 1000, persist_dir: str = "./chroma_db") -> None:
    print("Loading data...")
    df = load_data()

    print("Converting to text documents...")
    transactions, summaries = build_all_texts(df)
    print(f"  Transaction docs : {len(transactions):,}")
    print(f"  Summary docs     : {len(summaries):,}")

    print(f"Chunking (size={chunk_size})...")
    transaction_chunks = chunk_documents(transactions, chunk_size)
    summary_chunks = chunk_documents(summaries, chunk_size)
    print(f"  Transaction chunks : {len(transaction_chunks):,}")
    print(f"  Summary chunks     : {len(summary_chunks):,}")

    print("Initializing ChromaDB...")
    client = get_client(persist_dir)
    ef = get_embedding_function()

    print("Upserting transaction chunks (this may take a few minutes)...")
    t0 = time.time()
    txn_col = get_collection(client, "transactions", ef)
    upsert_chunks(txn_col, transaction_chunks)
    print(f"  Done in {time.time() - t0:.1f}s — {txn_col.count():,} docs in collection")

    print("Upserting summary chunks...")
    t0 = time.time()
    sum_col = get_collection(client, "summaries", ef)
    upsert_chunks(sum_col, summary_chunks)
    print(f"  Done in {time.time() - t0:.1f}s — {sum_col.count():,} docs in collection")

    print(f"\nIndex persisted to: {os.path.abspath(persist_dir)}")


def _smoke_test(persist_dir: str) -> None:
    client = get_client(persist_dir)
    ef = get_embedding_function()

    tests = [
        ("summaries", "total sales by year",        None),
        ("summaries", "West region profit margin",  {"region": "West"}),
        ("summaries", "Technology category revenue",{"category": "Technology"}),
        ("transactions", "high discount office supplies", None),
    ]

    print("\n── Smoke tests ──────────────────────────────────────────────────────")
    for col_name, q, where in tests:
        col = get_collection(client, col_name, ef)
        results = query(col, q, n_results=2, where=where)
        print(f"\nQ: '{q}' | collection={col_name} | filter={where}")
        for r in results:
            print(f"  [{r['id']}] (dist={r['distance']:.4f}) {r['text'][:120]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size",  type=int, default=1000)
    parser.add_argument("--persist-dir", type=str, default=_DEFAULT_DB)
    parser.add_argument("--smoke-test",  action="store_true",
                        help="Run sample queries after building the index")
    args = parser.parse_args()

    build_index(args.chunk_size, args.persist_dir)

    if args.smoke_test:
        _smoke_test(args.persist_dir)