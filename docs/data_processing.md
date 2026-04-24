# Data Processing Module

This document covers the three modules under `src/data_processing/` that transform the raw Superstore CSV into text documents ready for embedding and vector-database ingestion.

```
src/data_processing/
├── loader.py          # Load and inspect the raw dataset
├── text_converter.py  # Convert rows and aggregations into natural-language documents
└── chunker.py         # Split documents into fixed-size chunks for embedding
```

---

## loader.py

Responsible for reading the CSV file and providing a quick dataset overview.

### `load_data(path=None) → pd.DataFrame`

Loads the Superstore CSV into a Pandas DataFrame.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| None` | `None` | Absolute or relative path to the CSV file. When `None` the function resolves to `<repo_root>/data/Superstore.csv` automatically. |

**What it does:**
- Reads the file with `encoding="latin-1"` (required for the Superstore dataset).
- Parses `Order Date` and `Ship Date` as `datetime64` so downstream code can perform date arithmetic without manual conversion.

**Returns:** A DataFrame with all original columns plus properly typed date columns.

---

### `show_overview(df) → None`

Prints a structured console report covering:

| Section | Content |
|---|---|
| Basic Info | Row/column count, date range, missing value count |
| Columns & Types | Name and dtype of every column |
| Numeric Stats | `describe()` for Sales, Quantity, Discount, Profit |
| Category / Sub-Category / Region / Segment / Ship Mode | Value counts |
| Unique Counts | Cardinality of Customer ID, Product ID, City, State |
| Sample Rows | First three rows |

Used as a quick sanity-check when the dataset is first loaded.

---

## text_converter.py

Converts the tabular data into natural-language text documents.  
Each function returns a `list[dict]` where every element has the shape:

```python
{
    "id":       str,   # unique document identifier
    "text":     str,   # natural-language content fed to the embedding model
    "metadata": dict,  # key-value pairs stored alongside the vector for filtering
}
```

### Document types

#### 1. Transaction-level documents

**`transaction_to_text(row) → str`**

Converts a single DataFrame row into one English sentence describing the order.

Example output:
```
On 2017-11-08, customer Sean Miller (Consumer) from Philadelphia, Pennsylvania
(East region) ordered 2 unit(s) of 'Cisco TelePresence System EX90 Videoconferencing Unit'
(Category: Technology, Sub-Category: Machines) with a 20% discount.
Sales: $22,638.48, loss: $1,811.08. This transaction resulted in a loss.
Shipped via Second Class. This was a high-value order.
```

The sentence includes contextual annotations:
- `"resulted in a loss"` vs `"was profitable"` based on the `Profit` sign.
- Discount description omitted when 0%, labelled `"with a {n}% discount"` or `"with a heavy {n}% discount"` (≥ 40%).
- `"This was a high-value order."` appended when `Sales ≥ $500`.

**`build_transaction_texts(df) → list[dict]`**

Applies `transaction_to_text` to every row.  
`id` = `Row ID`; metadata keys: `order_date`, `year`, `month`, `region`, `category`, `sub_category`, `segment`, `state`.

---

#### 2. Monthly summaries

**`build_monthly_summaries(df) → list[dict]`**

One document per calendar month (e.g. `2016-03`).  
Each document reports: total sales, total profit, profit margin, unique order count, average discount.  
`id` format: `monthly_2016-03`; metadata keys: `type`, `year`, `month`.

---

#### 3. Annual summaries

**`build_annual_summaries(df) → list[dict]`**

One document per year (2014–2017).  
Each document includes: total sales, total profit, profit margin, unique order count, average discount, top revenue category, top region, and a year-over-year comparison (sales growth %, margin change in percentage points).  
`id` format: `annual_2016`; metadata keys: `type`, `year`.

---

#### 4. Category & sub-category summaries

**`build_category_summaries(df) → list[dict]`**

Two groups of documents:

- **Category level** (3 documents — Technology, Furniture, Office Supplies): total revenue, revenue rank among all categories, profit margin, transaction count, average discount, list of sub-categories.
- **Sub-category level** (17 documents): transaction count, total sales, profit margin, average discount.

`id` formats: `category_Technology`, `subcat_Technology_Copiers`.

---

#### 5. Regional summaries

**`build_regional_summaries(df) → list[dict]`**

Two groups:

- **Region level** (4 documents): transaction count, total sales, profit margin, number of states and cities covered, top 3 states by sales.
- **State level** (49 documents): transaction count, total sales, total profit.

`id` formats: `region_West`, `state_California`.

---

#### 6. Seasonal & quarterly summaries

**`build_seasonal_summaries(df) → list[dict]`**

Two groups:

- **Quarter × Year** (e.g. `2016 Q3`): total sales, profit margin, unique order count, best-selling category.
- **Season aggregate** (Winter / Spring / Summer / Fall across all years): total sales, profit margin, average discount, best-selling category.

Season mapping: Dec–Feb → Winter; Mar–May → Spring; Jun–Aug → Summer; Sep–Nov → Fall.  
Quarter mapping: Jan–Mar → Q1; Apr–Jun → Q2; Jul–Sep → Q3; Oct–Dec → Q4.

---

#### 7. Cross-dimension summaries

**`build_cross_dimension_summaries(df) → list[dict]`**

Pairwise breakdowns for the four most analytically useful dimension combinations:

| Combination | Example ID |
|---|---|
| Region × Category | `region_cat_West_Technology` |
| Year × Category | `year_cat_2016_Technology` |
| Year × Region | `year_region_2016_West` |
| Segment × Category | `segment_cat_Consumer_Technology` |

Each document reports transaction count, total sales, profit margin, average discount.

---

#### 8. Statistical summaries

**`build_statistical_summary(df) → list[dict]`**

Four thematic overview documents (all years combined):

| ID | Content |
|---|---|
| `stats_overview` | Total transactions, date span, geographic coverage, total revenue, average order value, category revenue ranking, highest-revenue month, top product |
| `stats_profit_analysis` | Total profit, overall margin, loss transaction rate, most/least profitable category and sub-category |
| `stats_discount_analysis` | Discounted transaction rate, average discount, high-discount (≥ 40%) transaction count and unprofitability rate, most-discounted sub-category |
| `stats_customer_analysis` | Unique customer count, top segment, top customer, repeat purchase rate, unique product count |

---

#### 9. Multi-year trend narrative

**`build_trend_summary(df) → list[dict]`**

A single document (`trend_annual_overview`) narrating year-over-year sales and profit-margin changes across 2014–2017.

---

#### 10. Comparative summaries

**`build_comparative_summaries(df) → list[dict]`**

Two pre-built comparison documents addressing the most common analytical questions:

| ID | Comparison |
|---|---|
| `compare_tech_vs_furniture` | Technology vs Furniture: annual sales per year, total revenue, profit margin for each |
| `compare_west_vs_east` | West vs East region: total sales, total profit, per-year profit margins for each |

---

#### 11. Seasonal & quarterly ranking

**`build_seasonal_ranking_summary(df) → list[dict]`**

One document (`seasonal_quarter_ranking`) ranking all four quarters and all four seasons by total sales and profit margin across all years.

---

#### 12. Region ranking

**`build_region_ranking_summary(df) → list[dict]`**

One document (`region_ranking`) ranking all four regions by total sales and total profit.

---

#### 13. Top performers ranking

**`build_top_performers_summary(df) → list[dict]`**

Three documents:

| ID | Content |
|---|---|
| `category_revenue_ranking` | All categories ranked by total revenue with profit margins |
| `top_states_by_sales` | Top 10 states by total revenue with profit margins |
| `subcategory_margin_ranking` | Top 5 highest-margin and bottom 3 lowest-margin sub-categories (minimum 50 transactions) |

---

### Entry points

| Function | Returns |
|---|---|
| `build_transaction_docs(df)` | Transaction documents only |
| `build_summary_docs(df)` | All 12 types of aggregated summary documents |
| `build_all_texts(df)` | `(transaction_docs, summary_docs)` as a tuple |

`build_summary_docs` produces roughly **300–400 documents** depending on the dataset; `build_transaction_docs` produces one document per transaction row (~9,994).

---

## chunker.py

Splits text documents into fixed-size character-level chunks before embedding.

### `Chunk` dataclass

```python
@dataclass
class Chunk:
    id:         str    # document ID, or "{id}_part{n}" when split
    text:       str    # chunk content
    metadata:   dict   # original metadata + "chunk_size" key
    chunk_size: int    # the target chunk size used
```

---

### `_split_text(text, chunk_size, overlap=50) → list[str]`

Internal helper. Returns the original string unchanged if `len(text) ≤ chunk_size`. Otherwise splits into overlapping windows:

```
window 1: text[0 : chunk_size]
window 2: text[chunk_size - overlap : 2*chunk_size - overlap]
...
```

The 50-character overlap preserves sentence continuity at boundaries.

---

### `chunk_documents(docs, chunk_size, overlap=50) → list[Chunk]`

Converts a list of raw documents into `Chunk` objects.

- Documents shorter than `chunk_size` become a single chunk (id unchanged).
- Longer documents are split; each part gets the id suffix `_part0`, `_part1`, etc.
- `"chunk_size"` is injected into every chunk's metadata for traceability.

In practice, most Superstore documents (transactions ~200 chars, summaries ~150–400 chars) fit entirely within any of the three tested sizes, so splitting rarely occurs.

---

### `chunk_all_sizes(docs) → dict[int, list[Chunk]]`

Convenience wrapper that calls `chunk_documents` for all three chunk sizes at once and returns a dictionary keyed by size.

```python
chunks = chunk_all_sizes(docs)
# chunks[500]  → list[Chunk]  (fine-grained)
# chunks[1000] → list[Chunk]  (balanced, recommended)
# chunks[2000] → list[Chunk]  (coarse-grained)
```

---

### Chunk size rationale

| Size | Characteristic | Trade-off |
|---|---|---|
| 500 | Fine-grained | Higher retrieval precision; narrow context per chunk |
| **1000** | **Balanced (default)** | **Best overall performance for this dataset** |
| 2000 | Coarse-grained | More context per chunk; may dilute relevance for specific queries |

Because the documents in this dataset are short by design, all three chunk sizes produce the same number of chunks (10,432) — no document is ever split. This is a deliberate outcome: each document was constructed to cover exactly one transaction, one time period, or one dimension combination, so every unit is already semantically atomic. The more consequential granularity decision was made in `text_converter.py` — representing each sales order as an individual document rather than concatenating multiple rows — which determines what the embedding model sees as a single retrievable unit. Chunk size remains configurable for future iterations where document length may grow.

---

### `print_chunk_stats(chunks_by_size) → None`

Prints a comparison table for a quick sanity check:

```
Chunk size comparison:
  Size   # Chunks   Avg len   Min len   Max len
--------------------------------------------------
   500     10,432       198       ...       ...
  1000     10,432       198       ...       ...
  2000     10,432       198       ...       ...
```

---

## Data flow summary

```
Superstore.csv
    │
    ▼  loader.load_data()
DataFrame
    │
    ├──▶  text_converter.build_transaction_docs()  ──▶  ~9,994 transaction dicts
    │
    └──▶  text_converter.build_summary_docs()      ──▶  ~350 summary dicts
              │
              └──▶  chunker.chunk_documents(docs, chunk_size=1000)
                        │
                        ▼
                   list[Chunk]  →  vector_db ingestion
```
