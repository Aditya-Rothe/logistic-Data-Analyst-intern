
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def investigate_outliers(df, column, dataset_name):
    values = df[column].dropna()

    if values.empty:
        print(f"\n{column}: No available values")
        return

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = values[
        (values < lower_bound) | (values > upper_bound)
    ]

    print(f"\nDataset: {dataset_name}")
    print(f"Column: {column}")
    print(f"Minimum: {values.min():.2f}")
    print(f"Maximum: {values.max():.2f}")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Potential outliers: {len(outliers)}")


# 1. Investigate order item values
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")

print("\n========== ORDER ITEMS OUTLIER ANALYSIS ==========")

for column in ["price", "freight_value"]:
    investigate_outliers(items, column, "Order Items")


# 2. Investigate product dimensions and weight
products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")

print("\n========== PRODUCT OUTLIER ANALYSIS ==========")

product_columns = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

for column in product_columns:
    investigate_outliers(products, column, "Products")


# 3. Calculate delivery duration and delay
orders = pd.read_csv(
    DATA_DIR / "olist_orders_dataset.csv",
    parse_dates=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
)

delivered = orders[
    orders["order_status"].eq("delivered")
].copy()

delivered["delivery_duration_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400

delivered["delivery_delay_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

print("\n========== DELIVERY OUTLIER ANALYSIS ==========")

for column in [
    "delivery_duration_days",
    "delivery_delay_days"
]:
    investigate_outliers(delivered, column, "Orders")


# 4. Check impossible negative values
print("\n========== NEGATIVE VALUE CHECK ==========")

for column in ["price", "freight_value"]:
    print(f"{column} negative values:", (items[column] < 0).sum())

for column in product_columns:
    print(f"{column} negative values:", (products[column] < 0).sum())

print("\nOutlier investigation completed!")
