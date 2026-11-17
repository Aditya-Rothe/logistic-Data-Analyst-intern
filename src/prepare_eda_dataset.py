
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

# 1. Load datasets
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")

items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")

customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")

# 2. Convert order date columns
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column],
        errors="coerce"
    )

# 3. Aggregate order items at order level
order_items_summary = (
    items.groupby("order_id")
    .agg(
        item_count=("order_item_id", "count"),
        total_product_price=("price", "sum"),
        total_freight=("freight_value", "sum")
    )
    .reset_index()
)

# 4. Merge orders with item summaries
eda_df = orders.merge(
    order_items_summary,
    on="order_id",
    how="left",
    validate="one_to_one"
)

# 5. Add customer location
customer_location = customers[
    ["customer_id", "customer_state"]
]

eda_df = eda_df.merge(
    customer_location,
    on="customer_id",
    how="left",
    validate="many_to_one"
)

# 6. Calculate logistics metrics
eda_df["total_order_value"] = (
    eda_df["total_product_price"]
    + eda_df["total_freight"]
)

# Calculate delivery duration in days
eda_df["delivery_duration_days"] = (
    eda_df["order_delivered_customer_date"]
    - eda_df["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Calculate delivery delay using calendar dates
actual_date = (
    eda_df["order_delivered_customer_date"].dt.normalize()
)

estimated_date = (
    eda_df["order_estimated_delivery_date"].dt.normalize()
)

eda_df["delivery_delay_days"] = (
    actual_date - estimated_date
).dt.days


# 7. Create delivery performance categories
eda_df["delivery_performance"] = "not delivered"

# Delivered orders
delivered_mask = eda_df["order_status"].eq("delivered")

# Delivered orders with valid actual and estimated dates
valid_delivery_mask = (
    delivered_mask
    & eda_df["delivery_delay_days"].notna()
)

# Delivered but delivery dates are unavailable
eda_df.loc[
    delivered_mask & ~valid_delivery_mask,
    "delivery_performance"
] = "date unavailable"

# On-time deliveries
eda_df.loc[
    valid_delivery_mask & (eda_df["delivery_delay_days"] <= 0),
    "delivery_performance"
] = "on time"

# Late deliveries
eda_df.loc[
    valid_delivery_mask & (eda_df["delivery_delay_days"] > 0),
    "delivery_performance"
] = "late"

# 8. Extract purchase month
eda_df["purchase_month"] = (
    eda_df["order_purchase_timestamp"].dt.to_period("M").astype(str)
)

# 9. Save analytical dataset
output_path = DATA_DIR / "olist_order_level_eda.csv"

eda_df.to_csv(output_path, index=False)

# 10. Display summary
print("\n========== EDA DATASET SUMMARY ==========")

print("Dataset shape:", eda_df.shape)
print("Unique orders:", eda_df["order_id"].nunique())
print("Duplicate order IDs:", eda_df["order_id"].duplicated().sum())

print("\nOrder status distribution:")
print(eda_df["order_status"].value_counts())

print("\nDelivery performance:")
print(eda_df["delivery_performance"].value_counts())

print("\nMissing values:")
print(eda_df.isna().sum()[eda_df.isna().sum() > 0])

print("\nSaved analytical dataset to:", output_path)

print("\nEDA dataset preparation completed!")
