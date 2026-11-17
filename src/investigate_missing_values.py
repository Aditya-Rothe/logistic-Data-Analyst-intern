
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

# Load orders dataset
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")

# 1. Check order status distribution
print("\nORDER STATUS DISTRIBUTION")
print(orders["order_status"].value_counts(dropna=False))

# 2. Missing delivery dates by order status
date_columns = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date"
]

print("\nMISSING DATES BY ORDER STATUS")

for column in date_columns:
    print(f"\n{column}")
    result = orders.groupby("order_status")[column].apply(
        lambda x: x.isna().sum()
    )
    print(result)

# 3. Investigate missing product information
products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")

print("\nPRODUCTS WITH MISSING VALUES")
print(products.isna().sum())

print("\nPRODUCTS MISSING WEIGHT OR DIMENSIONS")
dimension_columns = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

print(products[products[dimension_columns].isna().any(axis=1)])

# 4. Investigate geolocation duplicates
geo = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")

print("\nGEOLOCATION DUPLICATE ANALYSIS")
print("Total records:", len(geo))
print("Exact duplicate records:", geo.duplicated().sum())

print("\nSample exact duplicate records:")
print(geo[geo.duplicated(keep=False)].head(10))
