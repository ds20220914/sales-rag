# Embedding Model Comparison — Sales RAG

_Generated automatically by `compare_embeddings.py`_

## Model Overview

| Model | Params | Dims | Index time (s) | HIGH | MED | LOW | Avg top-1 dist |
|-------|--------|------|---------------|------|-----|-----|----------------|
| MiniLM-L6 (baseline) | 22 M | 384 | 89 | 13 | 0 | 0 | 0.2701 |
| multi-qa-MiniLM-L6 (QA-tuned) | 22 M | 384 | 90 | 13 | 0 | 0 | 0.2541 |
| mpnet-base-v2 (high-quality) | 110 M | 768 | 574 | 13 | 0 | 0 | 0.2656 |
| bge-small-en-v1.5 (SOTA small) | 33 M | 384 | 174 | 13 | 0 | 0 | 0.1796 |

## Per-Query Top-1 Distance

| # | Query | MiniLM-L6 | multi-qa-MiniLM-L6 | mpnet-base-v2 | bge-small-en-v1.5 |
|---|-------|-------|-------|-------|-------|
| 01 | Annual sales trend | 0.2713 [HIGH] | 0.2642 [HIGH] | 0.2798 [HIGH] | 0.1947 [HIGH] |
| 02 | Seasonal pattern | 0.2485 [HIGH] | 0.1495 [HIGH] | 0.1766 [HIGH] | 0.1565 [HIGH] |
| 03 | Profit margin change by year | 0.3394 [HIGH] | 0.2812 [HIGH] | 0.2953 [HIGH] | 0.1892 [HIGH] |
| 04 | Top revenue category | 0.2416 [HIGH] | 0.2034 [HIGH] | 0.2686 [HIGH] | 0.1820 [HIGH] |
| 05 | Highest margin sub-category | 0.2195 [HIGH] | 0.2029 [HIGH] | 0.2018 [HIGH] | 0.1420 [HIGH] |
| 06 | Discount pattern — Technology | 0.4067 [HIGH] | 0.4159 [HIGH] | 0.4226 [HIGH] | 0.2069 [HIGH] |
| 07 | Best performing region | 0.1997 [HIGH] | 0.1539 [HIGH] | 0.1851 [HIGH] | 0.1212 [HIGH] |
| 08 | West region performance | 0.1481 [HIGH] | 0.1798 [HIGH] | 0.2593 [HIGH] | 0.1531 [HIGH] |
| 09 | Top states by sales | 0.1948 [HIGH] | 0.1766 [HIGH] | 0.2392 [HIGH] | 0.1533 [HIGH] |
| 10 | Technology vs Furniture | 0.1587 [HIGH] | 0.2080 [HIGH] | 0.1729 [HIGH] | 0.1254 [HIGH] |
| 11 | West vs East profit | 0.1560 [HIGH] | 0.1473 [HIGH] | 0.1193 [HIGH] | 0.1107 [HIGH] |
| 12 | High-discount transactions | 0.4370 [HIGH] | 0.4234 [HIGH] | 0.3475 [HIGH] | 0.3161 [HIGH] |
| 13 | High-value Technology orders | 0.4904 [HIGH] | 0.4976 [HIGH] | 0.4841 [HIGH] | 0.2837 [HIGH] |

## Per-Query Top-1 Document Retrieved

| # | Query | MiniLM-L6 | multi-qa-MiniLM-L6 | mpnet-base-v2 | bge-small-en-v1.5 |
|---|-------|-------|-------|-------|-------|
| 01 | Annual sales trend | trend_annual_overview | trend_annual_overview | trend_annual_overview | trend_annual_overview |
| 02 | Seasonal pattern | seasonal_quarter_ranking | seasonal_quarter_ranking | seasonal_quarter_ranking | seasonal_quarter_ranking |
| 03 | Profit margin change by year | trend_annual_overview | trend_annual_overview | trend_annual_overview | trend_annual_overview |
| 04 | Top revenue category | category_revenue_ranking | category_revenue_ranking | category_revenue_ranking | category_revenue_ranking |
| 05 | Highest margin sub-category | subcategory_margin_ranking | subcategory_margin_ranking | subcategory_margin_ranking | subcategory_margin_ranking |
| 06 | Discount pattern — Technology | region_cat_Central_Technology | subcat_Technology_Accessories | segment_cat_Corporate_Technology | category_Technology |
| 07 | Best performing region | region_ranking | region_ranking | region_ranking | region_ranking |
| 08 | West region performance | year_region_2016_West | year_region_2017_West | year_region_2014_West | region_West |
| 09 | Top states by sales | top_states_by_sales | top_states_by_sales | top_states_by_sales | top_states_by_sales |
| 10 | Technology vs Furniture | compare_tech_vs_furniture | compare_tech_vs_furniture | compare_tech_vs_furniture | compare_tech_vs_furniture |
| 11 | West vs East profit | compare_west_vs_east | compare_west_vs_east | compare_west_vs_east | compare_west_vs_east |
| 12 | High-discount transactions | 1670 | 4003 | 5017 | 551 |
| 13 | High-value Technology orders | 4278 | 4177 | 5684 | 7552 |

## Analysis

**Best average top-1 distance**: bge-small-en-v1.5 (SOTA small) (0.1796)

**Fastest indexing**: MiniLM-L6 (baseline) (89s)

### Queries where top-1 document differs across models

| # | Query | Documents retrieved |
|---|-------|---------------------|
| 06 | Discount pattern — Technology | MiniLM-L6: `region_cat_Central_Technology` / multi-qa-MiniLM-L6: `subcat_Technology_Accessories` / mpnet-base-v2: `segment_cat_Corporate_Technology` / bge-small-en-v1.5: `category_Technology` |
| 08 | West region performance | MiniLM-L6: `year_region_2016_West` / multi-qa-MiniLM-L6: `year_region_2017_West` / mpnet-base-v2: `year_region_2014_West` / bge-small-en-v1.5: `region_West` |
| 12 | High-discount transactions | MiniLM-L6: `1670` / multi-qa-MiniLM-L6: `4003` / mpnet-base-v2: `5017` / bge-small-en-v1.5: `551` |
| 13 | High-value Technology orders | MiniLM-L6: `4278` / multi-qa-MiniLM-L6: `4177` / mpnet-base-v2: `5684` / bge-small-en-v1.5: `7552` |

### Queries where at least one model returned MED/LOW

| # | Query | MiniLM-L6 | multi-qa-MiniLM-L6 | mpnet-base-v2 | bge-small-en-v1.5 |
|---|-------|-------|-------|-------|-------|
_All models returned HIGH relevance on every query._
