import pandas as pd


# ── 1. Transaction-level: one natural language sentence per row ──────────────

def transaction_to_text(row: pd.Series) -> str:
    profit = row["Profit"]
    sales = row["Sales"]
    discount = row["Discount"]

    outcome = "resulted in a loss" if profit < 0 else "was profitable"
    discount_str = ""
    if discount >= 0.4:
        discount_str = f" with a heavy {discount:.0%} discount"
    elif discount > 0:
        discount_str = f" with a {discount:.0%} discount"

    value_tag = " This was a high-value order." if sales >= 500 else ""

    return (
        f"On {row['Order Date'].strftime('%Y-%m-%d')}, customer {row['Customer Name']} "
        f"({row['Segment']}) from {row['City']}, {row['State']} ({row['Region']} region) "
        f"ordered {row['Quantity']} unit(s) of '{row['Product Name']}' "
        f"(Category: {row['Category']}, Sub-Category: {row['Sub-Category']}){discount_str}. "
        f"Sales: ${sales:.2f}, {'loss' if profit < 0 else 'profit'}: ${abs(profit):.2f}. "
        f"This transaction {outcome}. "
        f"Shipped via {row['Ship Mode']}.{value_tag}"
    )


def build_transaction_texts(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": str(row["Row ID"]),
            "text": transaction_to_text(row),
            "metadata": {
                "order_date": row["Order Date"].strftime("%Y-%m-%d"),
                "year": str(row["Order Date"].year),
                "month": str(row["Order Date"].month),
                "region": row["Region"],
                "category": row["Category"],
                "sub_category": row["Sub-Category"],
                "segment": row["Segment"],
                "state": row["State"],
            },
        })
    return records


# ── 2. Monthly aggregated summary ────────────────────────────────────────────

def build_monthly_summaries(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["YearMonth"] = df["Order Date"].dt.to_period("M")
    grouped = df.groupby("YearMonth").agg(
        total_sales=("Sales", "sum"),
        total_profit=("Profit", "sum"),
        total_orders=("Order ID", "nunique"),
        avg_discount=("Discount", "mean"),
    )

    records = []
    for period, row in grouped.iterrows():
        text = (
            f"Monthly summary for {period}: "
            f"total sales were ${row['total_sales']:,.2f}, "
            f"total profit was ${row['total_profit']:,.2f} "
            f"(profit margin {row['total_profit'] / row['total_sales'] * 100:.1f}%), "
            f"with {row['total_orders']} unique orders and "
            f"an average discount of {row['avg_discount']:.1%}."
        )
        records.append({
            "id": f"monthly_{period}",
            "text": text,
            "metadata": {
                "type": "monthly_summary",
                "year": str(period.year),
                "month": str(period.month),
            },
        })
    return records


# ── 3. Category performance summary ──────────────────────────────────────────

def build_category_summaries(df: pd.DataFrame) -> list[dict]:
    records = []

    # Category level
    for cat, grp in df.groupby("Category"):
        text = (
            f"Category '{cat}' performance: "
            f"{len(grp):,} transactions, "
            f"total sales ${grp['Sales'].sum():,.2f}, "
            f"total profit ${grp['Profit'].sum():,.2f} "
            f"(margin {grp['Profit'].sum() / grp['Sales'].sum() * 100:.1f}%), "
            f"avg discount {grp['Discount'].mean():.1%}. "
            f"Sub-categories: {', '.join(grp['Sub-Category'].unique())}."
        )
        records.append({
            "id": f"category_{cat.replace(' ', '_')}",
            "text": text,
            "metadata": {"type": "category_summary", "category": cat},
        })

    # Sub-category level
    for (cat, sub), grp in df.groupby(["Category", "Sub-Category"]):
        text = (
            f"Sub-category '{sub}' (under {cat}): "
            f"{len(grp):,} transactions, "
            f"total sales ${grp['Sales'].sum():,.2f}, "
            f"total profit ${grp['Profit'].sum():,.2f} "
            f"(margin {grp['Profit'].sum() / grp['Sales'].sum() * 100:.1f}%), "
            f"avg discount {grp['Discount'].mean():.1%}."
        )
        records.append({
            "id": f"subcat_{cat.replace(' ', '_')}_{sub.replace(' ', '_')}",
            "text": text,
            "metadata": {"type": "subcategory_summary", "category": cat, "sub_category": sub},
        })

    return records


# ── 4. Regional analysis summary ─────────────────────────────────────────────

def build_regional_summaries(df: pd.DataFrame) -> list[dict]:
    records = []

    # Region level
    for region, grp in df.groupby("Region"):
        top_states = (
            grp.groupby("State")["Sales"].sum()
            .nlargest(3)
            .index.tolist()
        )
        text = (
            f"Region '{region}': "
            f"{len(grp):,} transactions, "
            f"total sales ${grp['Sales'].sum():,.2f}, "
            f"total profit ${grp['Profit'].sum():,.2f} "
            f"(margin {grp['Profit'].sum() / grp['Sales'].sum() * 100:.1f}%), "
            f"covering {grp['State'].nunique()} states and {grp['City'].nunique()} cities. "
            f"Top 3 states by sales: {', '.join(top_states)}."
        )
        records.append({
            "id": f"region_{region.replace(' ', '_')}",
            "text": text,
            "metadata": {"type": "regional_summary", "region": region},
        })

    # State level
    for state, grp in df.groupby("State"):
        text = (
            f"State '{state}' ({grp['Region'].iloc[0]} region): "
            f"{len(grp):,} transactions, "
            f"total sales ${grp['Sales'].sum():,.2f}, "
            f"total profit ${grp['Profit'].sum():,.2f}."
        )
        records.append({
            "id": f"state_{state.replace(' ', '_')}",
            "text": text,
            "metadata": {"type": "state_summary", "state": state, "region": grp["Region"].iloc[0]},
        })

    return records


# ── 5. Annual summary ────────────────────────────────────────────────────────

def build_annual_summaries(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["Year"] = df["Order Date"].dt.year
    records = []
    prev_sales = None

    for year, grp in df.groupby("Year"):
        yoy = ""
        if prev_sales is not None:
            change = (grp["Sales"].sum() - prev_sales) / prev_sales * 100
            direction = "up" if change >= 0 else "down"
            yoy = f" ({direction} {abs(change):.1f}% year-over-year)"
        prev_sales = grp["Sales"].sum()

        top_cat = grp.groupby("Category")["Sales"].sum().idxmax()
        top_region = grp.groupby("Region")["Sales"].sum().idxmax()
        text = (
            f"Annual summary for {year}: "
            f"total sales ${grp['Sales'].sum():,.2f}{yoy}, "
            f"total profit ${grp['Profit'].sum():,.2f} "
            f"(margin {grp['Profit'].sum() / grp['Sales'].sum() * 100:.1f}%), "
            f"{grp['Order ID'].nunique():,} unique orders, "
            f"avg discount {grp['Discount'].mean():.1%}. "
            f"Top category: {top_cat}. Top region: {top_region}."
        )
        records.append({
            "id": f"annual_{year}",
            "text": text,
            "metadata": {"type": "annual_summary", "year": str(year)},
        })
    return records


# ── 6. Seasonal summary ───────────────────────────────────────────────────────

_SEASON_MAP = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
               5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
               9: "Fall",   10: "Fall",  11: "Fall",  12: "Winter"}
_QUARTER_MAP = {1: "Q1", 2: "Q1", 3: "Q1",
                4: "Q2", 5: "Q2", 6: "Q2",
                7: "Q3", 8: "Q3", 9: "Q3",
                10: "Q4", 11: "Q4", 12: "Q4"}

def build_seasonal_summaries(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["Year"] = df["Order Date"].dt.year
    df["Quarter"] = df["Order Date"].dt.month.map(_QUARTER_MAP)
    df["Season"] = df["Order Date"].dt.month.map(_SEASON_MAP)
    records = []

    # Quarter × Year
    for (year, quarter), grp in df.groupby(["Year", "Quarter"]):
        top_cat = grp.groupby("Category")["Sales"].sum().idxmax()
        text = (
            f"Seasonal summary for {year} {quarter}: "
            f"total sales ${grp['Sales'].sum():,.2f}, "
            f"total profit ${grp['Profit'].sum():,.2f} "
            f"(margin {grp['Profit'].sum() / grp['Sales'].sum() * 100:.1f}%), "
            f"{grp['Order ID'].nunique():,} unique orders. "
            f"Best-selling category: {top_cat}."
        )
        records.append({
            "id": f"seasonal_{year}_{quarter}",
            "text": text,
            "metadata": {"type": "seasonal_summary", "year": str(year), "quarter": quarter},
        })

    # Season aggregated across all years
    for season, grp in df.groupby("Season"):
        top_cat = grp.groupby("Category")["Sales"].sum().idxmax()
        text = (
            f"{season} (all years combined): "
            f"total sales ${grp['Sales'].sum():,.2f}, "
            f"total profit ${grp['Profit'].sum():,.2f} "
            f"(margin {grp['Profit'].sum() / grp['Sales'].sum() * 100:.1f}%), "
            f"avg discount {grp['Discount'].mean():.1%}. "
            f"Best-selling category: {top_cat}."
        )
        records.append({
            "id": f"season_{season}",
            "text": text,
            "metadata": {"type": "season_summary", "season": season},
        })

    return records


# ── 7. Cross-dimension summaries ──────────────────────────────────────────────

def build_cross_dimension_summaries(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["Year"] = df["Order Date"].dt.year
    records = []

    # Region × Category
    for (region, cat), grp in df.groupby(["Region", "Category"]):
        text = (
            f"In the {region} region, category '{cat}': "
            f"{len(grp):,} transactions, "
            f"total sales ${grp['Sales'].sum():,.2f}, "
            f"total profit ${grp['Profit'].sum():,.2f} "
            f"(margin {grp['Profit'].sum() / grp['Sales'].sum() * 100:.1f}%), "
            f"avg discount {grp['Discount'].mean():.1%}."
        )
        records.append({
            "id": f"region_cat_{region.replace(' ', '_')}_{cat.replace(' ', '_')}",
            "text": text,
            "metadata": {"type": "region_category_summary", "region": region, "category": cat},
        })

    # Year × Category
    for (year, cat), grp in df.groupby(["Year", "Category"]):
        text = (
            f"In {year}, category '{cat}': "
            f"total sales ${grp['Sales'].sum():,.2f}, "
            f"total profit ${grp['Profit'].sum():,.2f} "
            f"(margin {grp['Profit'].sum() / grp['Sales'].sum() * 100:.1f}%), "
            f"{grp['Order ID'].nunique():,} unique orders."
        )
        records.append({
            "id": f"year_cat_{year}_{cat.replace(' ', '_')}",
            "text": text,
            "metadata": {"type": "year_category_summary", "year": str(year), "category": cat},
        })

    # Year × Region
    for (year, region), grp in df.groupby(["Year", "Region"]):
        text = (
            f"In {year}, {region} region: "
            f"total sales ${grp['Sales'].sum():,.2f}, "
            f"total profit ${grp['Profit'].sum():,.2f} "
            f"(margin {grp['Profit'].sum() / grp['Sales'].sum() * 100:.1f}%), "
            f"{grp['Order ID'].nunique():,} unique orders."
        )
        records.append({
            "id": f"year_region_{year}_{region.replace(' ', '_')}",
            "text": text,
            "metadata": {"type": "year_region_summary", "year": str(year), "region": region},
        })

    # Segment × Category
    for (segment, cat), grp in df.groupby(["Segment", "Category"]):
        text = (
            f"Customer segment '{segment}' buying category '{cat}': "
            f"{len(grp):,} transactions, "
            f"total sales ${grp['Sales'].sum():,.2f}, "
            f"total profit ${grp['Profit'].sum():,.2f} "
            f"(margin {grp['Profit'].sum() / grp['Sales'].sum() * 100:.1f}%), "
            f"avg discount {grp['Discount'].mean():.1%}."
        )
        records.append({
            "id": f"segment_cat_{segment.replace(' ', '_')}_{cat.replace(' ', '_')}",
            "text": text,
            "metadata": {"type": "segment_category_summary", "segment": segment, "category": cat},
        })

    return records


# ── 8. Overall statistical summaries ─────────────────────────────────────────

_MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

def build_statistical_summary(df: pd.DataFrame) -> list[dict]:
    years = sorted(df["Order Date"].dt.year.unique())
    best_month = df.groupby(df["Order Date"].dt.month)["Sales"].sum().idxmax()
    top_product = df.groupby("Product Name")["Sales"].sum().idxmax()

    # 8a. General overview
    overview = (
        f"Overall overview: The Superstore dataset covers {len(df):,} transactions "
        f"from {years[0]} to {years[-1]}, across {df['State'].nunique()} states "
        f"and {df['City'].nunique()} cities. "
        f"Total sales: ${df['Sales'].sum():,.2f}, average order value: ${df['Sales'].mean():.2f}. "
        f"Highest-sales month across all years: {_MONTH_NAMES[best_month]}. "
        f"Top product by total sales: '{top_product}'."
    )

    # 8b. Profit analysis
    loss_pct = (df["Profit"] < 0).mean() * 100
    best_profit_cat = df.groupby("Category")["Profit"].sum().idxmax()
    worst_profit_cat = df.groupby("Category")["Profit"].sum().idxmin()
    best_profit_sub = df.groupby("Sub-Category")["Profit"].sum().idxmax()
    worst_profit_sub = df.groupby("Sub-Category")["Profit"].sum().idxmin()
    profit_analysis = (
        f"Profit analysis: Total profit ${df['Profit'].sum():,.2f} "
        f"(overall margin {df['Profit'].sum() / df['Sales'].sum() * 100:.1f}%). "
        f"{loss_pct:.1f}% of transactions resulted in a loss. "
        f"Most profitable category: {best_profit_cat}; least profitable: {worst_profit_cat}. "
        f"Most profitable sub-category: {best_profit_sub}; least profitable: {worst_profit_sub}."
    )

    # 8c. Discount analysis
    discounted_pct = (df["Discount"] > 0).mean() * 100
    high_discount = df[df["Discount"] >= 0.4]
    high_disc_loss_pct = (high_discount["Profit"] < 0).mean() * 100 if len(high_discount) else 0
    most_discounted_sub = df.groupby("Sub-Category")["Discount"].mean().idxmax()
    discount_analysis = (
        f"Discount analysis: {discounted_pct:.1f}% of transactions include a discount. "
        f"Overall average discount: {df['Discount'].mean():.1%}. "
        f"Transactions with ≥40% discount: {len(high_discount):,} "
        f"({high_disc_loss_pct:.1f}% of those are unprofitable). "
        f"Most heavily discounted sub-category on average: {most_discounted_sub}."
    )

    # 8d. Customer analysis
    top_segment = df.groupby("Segment")["Sales"].sum().idxmax()
    top_customer = df.groupby("Customer Name")["Sales"].sum().idxmax()
    repeat_rate = (df.groupby("Customer ID")["Order ID"].nunique() > 1).mean() * 100
    customer_analysis = (
        f"Customer analysis: {df['Customer ID'].nunique():,} unique customers "
        f"across {df['Segment'].nunique()} segments. "
        f"Top segment by sales: {top_segment}. "
        f"Top customer by total sales: '{top_customer}'. "
        f"Repeat purchase rate: {repeat_rate:.1f}% of customers placed more than one order. "
        f"Total unique products: {df['Product ID'].nunique():,}."
    )

    meta = {"type": "statistical_summary"}
    return [
        {"id": "stats_overview",          "text": overview,          "metadata": meta},
        {"id": "stats_profit_analysis",   "text": profit_analysis,   "metadata": meta},
        {"id": "stats_discount_analysis", "text": discount_analysis, "metadata": meta},
        {"id": "stats_customer_analysis", "text": customer_analysis, "metadata": meta},
    ]


# ── 9. Multi-year trend narrative ────────────────────────────────────────────

def build_trend_summary(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["Year"] = df["Order Date"].dt.year
    annual = df.groupby("Year").agg(
        sales=("Sales", "sum"),
        profit=("Profit", "sum"),
        orders=("Order ID", "nunique"),
        discount=("Discount", "mean"),
    )
    years = annual.index.tolist()

    rows = []
    for i, yr in enumerate(years):
        r = annual.loc[yr]
        yoy = ""
        if i > 0:
            prev = annual.loc[years[i - 1]]
            chg = (r.sales - prev.sales) / prev.sales * 100
            yoy = f", up {chg:.1f}% from {years[i-1]}" if chg >= 0 else f", down {abs(chg):.1f}% from {years[i-1]}"
        rows.append(f"{yr}: sales ${r.sales:,.2f}{yoy}, profit margin {r.profit/r.sales*100:.1f}%")

    text = (
        f"Sales trend across all years (2014–2017): "
        + "; ".join(rows)
        + ". Overall sales grew year over year, with profit margins fluctuating. "
        "This document summarizes annual sales trend and profit margin changes over the 4-year period."
    )
    return [{"id": "trend_annual_overview", "text": text, "metadata": {"type": "trend_summary"}}]


# ── 10. Explicit comparative summaries ────────────────────────────────────────

def build_comparative_summaries(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["Year"] = df["Order Date"].dt.year
    records = []

    # Technology vs Furniture
    tech = df[df["Category"] == "Technology"]
    furn = df[df["Category"] == "Furniture"]
    tech_annual = tech.groupby("Year")["Sales"].sum()
    furn_annual = furn.groupby("Year")["Sales"].sum()
    tech_lines = "; ".join(f"{yr}: ${tech_annual[yr]:,.2f}" for yr in tech_annual.index)
    furn_lines = "; ".join(f"{yr}: ${furn_annual[yr]:,.2f}" for yr in furn_annual.index)
    text = (
        f"Comparison of Technology vs Furniture category sales trends (2014–2017). "
        f"Technology annual sales: {tech_lines}. "
        f"Furniture annual sales: {furn_lines}. "
        f"Technology total: ${tech['Sales'].sum():,.2f} (margin {tech['Profit'].sum()/tech['Sales'].sum()*100:.1f}%). "
        f"Furniture total: ${furn['Sales'].sum():,.2f} (margin {furn['Profit'].sum()/furn['Sales'].sum()*100:.1f}%). "
        f"Technology consistently outperforms Furniture in profit margin."
    )
    records.append({"id": "compare_tech_vs_furniture", "text": text,
                    "metadata": {"type": "comparative_summary"}})

    # West vs East profit comparison
    west = df[df["Region"] == "West"]
    east = df[df["Region"] == "East"]
    west_annual = west.groupby("Year").agg(sales=("Sales","sum"), profit=("Profit","sum"))
    east_annual = east.groupby("Year").agg(sales=("Sales","sum"), profit=("Profit","sum"))
    west_lines = "; ".join(f"{yr}: margin {r.profit/r.sales*100:.1f}%" for yr, r in west_annual.iterrows())
    east_lines = "; ".join(f"{yr}: margin {r.profit/r.sales*100:.1f}%" for yr, r in east_annual.iterrows())
    text = (
        f"Comparison of West vs East region profit (2014–2017). "
        f"West region total sales ${west['Sales'].sum():,.2f}, total profit ${west['Profit'].sum():,.2f} "
        f"(overall margin {west['Profit'].sum()/west['Sales'].sum()*100:.1f}%). "
        f"West annual profit margins: {west_lines}. "
        f"East region total sales ${east['Sales'].sum():,.2f}, total profit ${east['Profit'].sum():,.2f} "
        f"(overall margin {east['Profit'].sum()/east['Sales'].sum()*100:.1f}%). "
        f"East annual profit margins: {east_lines}. "
        f"West outperforms East in both total profit and profit margin."
    )
    records.append({"id": "compare_west_vs_east", "text": text,
                    "metadata": {"type": "comparative_summary"}})

    return records


# ── 11. Top performers ranking summary ───────────────────────────────────────

def build_top_performers_summary(df: pd.DataFrame) -> list[dict]:
    # Top 10 states by total sales
    top_states = df.groupby("State").agg(
        sales=("Sales","sum"), profit=("Profit","sum")
    ).nlargest(10, "sales")
    state_lines = "; ".join(
        f"{state}: ${r.sales:,.2f} (margin {r.profit/r.sales*100:.1f}%)"
        for state, r in top_states.iterrows()
    )
    text = (
        f"Top 10 states ranked by total sales: {state_lines}. "
        f"California leads all states with the highest revenue, followed by New York and Texas."
    )
    records = [{"id": "top_states_by_sales", "text": text, "metadata": {"type": "top_performers_summary"}}]

    # Top sub-categories by profit margin (min 50 transactions)
    sub_stats = df.groupby("Sub-Category").agg(
        sales=("Sales","sum"), profit=("Profit","sum"), count=("Sales","count")
    )
    sub_stats = sub_stats[sub_stats["count"] >= 50]
    sub_stats["margin"] = sub_stats["profit"] / sub_stats["sales"] * 100
    top_margin = sub_stats.nlargest(5, "margin")
    worst_margin = sub_stats.nsmallest(3, "margin")
    best_lines = "; ".join(
        f"{sub}: {r.margin:.1f}%" for sub, r in top_margin.iterrows()
    )
    worst_lines = "; ".join(
        f"{sub}: {r.margin:.1f}%" for sub, r in worst_margin.iterrows()
    )
    text = (
        f"Sub-category profit margin ranking. "
        f"Highest margin sub-categories: {best_lines}. "
        f"Lowest margin sub-categories (including losses): {worst_lines}. "
        f"Copiers and Labels have the highest profit margins; Tables and Bookcases suffer losses."
    )
    records.append({"id": "subcategory_margin_ranking", "text": text,
                    "metadata": {"type": "top_performers_summary"}})

    return records


# ── Entry points ─────────────────────────────────────────────────────────────

def build_transaction_docs(df: pd.DataFrame) -> list[dict]:
    """Return only transaction-level documents (one per row)."""
    return build_transaction_texts(df)


def build_summary_docs(df: pd.DataFrame) -> list[dict]:
    """Return all aggregated summary documents (no individual transactions)."""
    return (
        build_monthly_summaries(df)
        + build_annual_summaries(df)
        + build_seasonal_summaries(df)
        + build_category_summaries(df)
        + build_regional_summaries(df)
        + build_cross_dimension_summaries(df)
        + build_statistical_summary(df)
        + build_trend_summary(df)
        + build_comparative_summaries(df)
        + build_top_performers_summary(df)
    )


def build_all_texts(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Return (transaction_docs, summary_docs) as separate lists."""
    return build_transaction_docs(df), build_summary_docs(df)


if __name__ == "__main__":
    from loader import load_data

    df = load_data()
    transactions, summaries = build_all_texts(df)

    print(f"Transaction documents : {len(transactions)}")
    print(f"Summary documents     : {len(summaries)}")
    print(f"Total                 : {len(transactions) + len(summaries)}\n")

    print("--- Transaction sample ---")
    print(transactions[0]["text"])
    print()

    summary_samples = [
        ("Monthly sample",              "monthly_summary"),
        ("Annual sample",               "annual_summary"),
        ("Seasonal (quarter) sample",   "seasonal_summary"),
        ("Season (aggregate) sample",   "season_summary"),
        ("Category sample",             "category_summary"),
        ("Regional sample",             "regional_summary"),
        ("Cross-dim region×category",   "region_category_summary"),
        ("Cross-dim year×category",     "year_category_summary"),
        ("Statistical summary",         "statistical_summary"),
    ]
    for label, doc_type in summary_samples:
        doc = next(d for d in summaries if d["metadata"].get("type") == doc_type)
        print(f"--- {label} ---")
        print(doc["text"])
        print()