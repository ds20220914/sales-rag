import pandas as pd


def load_data(path: str = "data/Superstore.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    return df


def show_overview(df: pd.DataFrame) -> None:
    sep = "-" * 50

    print("=== Basic Info ===")
    print(f"Rows: {df.shape[0]:,}    Columns: {df.shape[1]}")
    print(f"Date range: {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    print(f"\n{sep}\n=== Columns & Types ===")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<20} {dtype}")

    print(f"\n{sep}\n=== Numeric Stats ===")
    print(df[["Sales", "Quantity", "Discount", "Profit"]].describe().round(2).to_string())

    print(f"\n{sep}\n=== Category Distribution ===")
    print(df["Category"].value_counts().to_string())

    print(f"\n{sep}\n=== Sub-Category Distribution ===")
    print(df["Sub-Category"].value_counts().to_string())

    print(f"\n{sep}\n=== Region Distribution ===")
    print(df["Region"].value_counts().to_string())

    print(f"\n{sep}\n=== Segment Distribution ===")
    print(df["Segment"].value_counts().to_string())

    print(f"\n{sep}\n=== Ship Mode Distribution ===")
    print(df["Ship Mode"].value_counts().to_string())

    print(f"\n{sep}\n=== Unique Counts ===")
    for col in ["Customer ID", "Product ID", "City", "State"]:
        print(f"  {col:<20} {df[col].nunique()}")

    print(f"\n{sep}\n=== Sample Rows ===")
    print(df.head(3).to_string())


if __name__ == "__main__":
    df = load_data()
    show_overview(df)