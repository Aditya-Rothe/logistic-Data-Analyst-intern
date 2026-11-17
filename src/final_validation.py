
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

print("\n========== FINAL DATA VALIDATION ==========")

# Check all raw and processed CSV files
for raw_file in sorted(RAW_DIR.glob("*.csv")):

    processed_file = PROCESSED_DIR / raw_file.name

    print(f"\nDataset: {raw_file.name}")

    if not processed_file.exists():
        print("ERROR: Processed file not found!")
        continue

    raw_df = pd.read_csv(raw_file)
    processed_df = pd.read_csv(processed_file)

    print("Raw rows:", len(raw_df))
    print("Processed rows:", len(processed_df))
    print("Rows changed:", len(raw_df) - len(processed_df))
    print("Missing values:", processed_df.isna().sum().sum())

    if raw_file.name == "olist_geolocation_dataset.csv":
        print(
            "Remaining exact duplicates:",
            processed_df.duplicated().sum()
        )

    if raw_file.name == "olist_products_dataset.csv":
        print(
            "Missing product weights:",
            processed_df["product_weight_g"].isna().sum()
        )

        print(
            "Zero product weights:",
            (processed_df["product_weight_g"] == 0).sum()
        )

        print(
            "Missing product categories:",
            processed_df["product_category_name"].isna().sum()
        )

    if raw_file.name == "olist_orders_dataset.csv":
        date_columns = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]

        print("\nDate conversion validation:")

        for column in date_columns:
            parsed = pd.to_datetime(
                processed_df[column],
                errors="coerce"
            )

            invalid = (
                processed_df[column].notna() & parsed.isna()
            ).sum()

            print(f"{column}: {invalid} invalid dates")

print("\n========== VALIDATION COMPLETED ==========")
