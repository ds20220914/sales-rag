"""
Manual retrieval test — runs a set of representative queries and prints results.

Usage:
    python src/vector_db/retrieval_test.py [--persist-dir ./chroma_db] [--n 5]
"""

import argparse
import sys
import os

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vector_db.store import get_client, get_embedding_function, get_collection, query

# ---------------------------------------------------------------------------
# Test cases: (label, collection, query_text, where_filter)
# ---------------------------------------------------------------------------
TEST_CASES = [
    # Trend
    ("Annual sales trend",
     "summaries", "sales trend over the years", None),
    ("Seasonal pattern",
     "summaries", "which season or quarter has the highest sales", None),
    ("Profit margin change by year",
     "summaries", "profit margin change over years", None),

    # Category
    ("Top revenue category",
     "summaries", "which category generates the most revenue", None),
    ("Highest margin sub-category",
     "summaries", "sub-category with highest profit margin", None),
    ("Discount pattern — Technology",
     "summaries", "discount rate in Technology category",
     {"category": "Technology"}),

    # Regional
    ("Best performing region",
     "summaries", "which region has the best sales and profit", None),
    ("West region performance",
     "summaries", "West region total sales and profit margin",
     {"region": "West"}),
    ("Top states by sales",
     "summaries", "top states ranked by total sales", None),

    # Comparative
    ("Technology vs Furniture",
     "summaries", "compare Technology and Furniture sales trends", None),
    ("West vs East profit",
     "summaries", "West region vs East region profit comparison", None),

    # Transaction-level
    ("High-discount transactions",
     "transactions", "large discount resulted in a loss", None),
    ("High-value Technology orders",
     "transactions", "expensive Technology product order high sales",
     {"category": "Technology"}),
]


def run_tests(persist_dir: str, n_results: int) -> None:
    client = get_client(persist_dir)
    ef = get_embedding_function()
    collections = {
        "transactions": get_collection(client, "transactions", ef),
        "summaries":    get_collection(client, "summaries",    ef),
    }

    passed = 0
    for i, (label, col_name, q, where) in enumerate(TEST_CASES, 1):
        col = collections[col_name]
        results = query(col, q, n_results=n_results, where=where)

        filter_str = f"  filter: {where}" if where else ""
        print(f"\n{'='*70}")
        print(f"[{i:02d}] {label}")
        print(f"  query : \"{q}\"")
        print(f"  source: {col_name}{filter_str}")
        print(f"  hits  : {len(results)}")
        print()

        for rank, r in enumerate(results, 1):
            dist = r["distance"]
            relevance = "HIGH" if dist < 0.5 else "MED" if dist < 0.9 else "LOW"
            snippet = r['text'][:200].strip().encode("ascii", errors="replace").decode("ascii")
            print(f"  #{rank} [{relevance}] dist={dist:.4f}  id={r['id']}")
            print(f"      {snippet}")
            if rank < len(results):
                print()

        if results and results[0]["distance"] < 0.9:
            passed += 1

    print(f"\n{'='*70}")
    print(f"Summary: {passed}/{len(TEST_CASES)} queries returned a high/medium relevance top result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--n", type=int, default=3, help="Results per query")
    args = parser.parse_args()

    run_tests(args.persist_dir, args.n)