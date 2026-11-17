
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Date columns that need conversion
date_columns = {
    "olist_orders_dataset.csv": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ],
    "olist_order_items_dataset.csv": [
        "shipping_limit_date"
    ],
    "olist_order_reviews_dataset.csv": [
        "review_creation_date",
        "review_answer_timestamp"
    ]
}

# Process every CSV file
for file_path in sorted(RAW_DIR.glob("*.csv")):

    print(f"\nProcessing: {file_path.name}")

    df = pd.read_csv(file_path)
    original_rows = len(df)

    # 1. Remove exact duplicate rows from geolocation
    if file_path.name == "olist_geolocation_dataset.csv":

        before = len(df)
        df = df.drop_duplicates()

        print("Duplicate rows removed:", before - len(df))

    # 2. Handle product data quality issues
    if file_path.name == "olist_products_dataset.csv":

        # Replace missing product categories
        df["product_category_name"] = (
            df["product_category_name"].fillna("unknown")
        )

        # Convert zero product weights into missing values
        zero_weights = (df["product_weight_g"] == 0).sum()

        df["product_weight_g"] = df["product_weight_g"].replace(
            0, float("nan")
        )

        print("Zero weights converted to missing:", zero_weights)
        print(
            "Remaining zero weights:",
            (df["product_weight_g"] == 0).sum()
        )

    # 3. Convert date columns and report invalid conversions
    columns_to_convert = date_columns.get(file_path.name, [])

    for column in columns_to_convert:

        original_non_missing = df[column].notna().sum()

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        invalid_dates = (
            original_non_missing - df[column].notna().sum()
        )

        print(f"{column} - invalid dates:", invalid_dates)

    # 4. Save processed dataset
    output_path = PROCESSED_DIR / file_path.name

    df.to_csv(output_path, index=False)

    # 5. Validation summary
    print("Original rows:", original_rows)
    print("Processed rows:", len(df))
    print("Rows removed:", original_rows - len(df))
    print("Remaining missing values:", df.isna().sum().sum())

    missing_values = df.isna().sum()

    if missing_values.sum() > 0:
        print("\nMissing values by column:")
        print(missing_values[missing_values > 0])

    print("Saved to:", output_path)

print("\nData cleaning and validation completed!")
