
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

# 1. Load products
products = pd.read_csv(
    DATA_DIR / "olist_products_dataset.csv"
)

print("\n========== ZERO PRODUCT VALUES ==========")

product_columns = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

for column in product_columns:
    print(f"{column}: {(products[column] == 0).sum()}")

# 2. Load order items
items = pd.read_csv(
    DATA_DIR / "olist_order_items_dataset.csv"
)

print("\n========== ZERO ORDER ITEM VALUES ==========")

for column in ["price", "freight_value"]:
    print(f"{column}: {(items[column] == 0).sum()}")

# 3. Load orders
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

print("\n========== FIVE LONGEST DELIVERY DURATIONS ==========")

print(
    delivered.nlargest(5, "delivery_duration_days")[
        [
            "order_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "delivery_duration_days"
        ]
    ].to_string(index=False)
)

print("\n========== FIVE LARGEST DELIVERY DELAYS ==========")

print(
    delivered.nlargest(5, "delivery_delay_days")[
        [
            "order_id",
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "order_delivered_customer_date",
            "delivery_delay_days"
        ]
    ].to_string(index=False)
)
