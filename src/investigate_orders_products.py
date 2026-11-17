
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

# Orders investigation
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")

print("\n--- ORDER STATUS DISTRIBUTION ---")
print(orders["order_status"].value_counts(dropna=False))

date_columns = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date"
]

print("\n--- MISSING DATES BY ORDER STATUS ---")

for column in date_columns:
    print(f"\n{column}")
    print(
        orders.groupby("order_status")[column]
        .apply(lambda x: x.isna().sum())
    )

# Products investigation
products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")

print("\n--- MISSING PRODUCT VALUES ---")
print(products.isna().sum())

dimension_columns = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

print("\n--- PRODUCTS WITH MISSING DIMENSIONS ---")
print(products.loc[
    products[dimension_columns].isna().any(axis=1)
])
