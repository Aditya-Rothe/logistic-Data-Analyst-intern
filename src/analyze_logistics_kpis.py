
from pathlib import Path
import pandas as pd

# 1. Load the prepared EDA dataset
DATA_PATH = Path("data/processed/olist_order_level_eda.csv")
df = pd.read_csv(DATA_PATH)

# 2. Identify valid delivered orders
delivered = df[df["order_status"] == "delivered"].copy()

valid_deliveries = delivered[
    delivered["delivery_delay_days"].notna()
].copy()

# 3. Calculate delivery KPIs
total_orders = len(df)
total_delivered = len(delivered)
valid_delivery_count = len(valid_deliveries)

on_time_count = (
    valid_deliveries["delivery_delay_days"] <= 0
).sum()

late_count = (
    valid_deliveries["delivery_delay_days"] > 0
).sum()

on_time_rate = on_time_count / valid_delivery_count * 100
late_rate = late_count / valid_delivery_count * 100

avg_delivery_days = delivered["delivery_duration_days"].mean()
median_delivery_days = delivered["delivery_duration_days"].median()

# 4. Calculate freight and order value metrics
orders_with_items = df[df["total_freight"].notna()]

avg_freight = orders_with_items["total_freight"].mean()
total_freight = orders_with_items["total_freight"].sum()

avg_product_value = orders_with_items["total_product_price"].mean()
avg_order_value = orders_with_items["total_order_value"].mean()

# 5. Display KPI results
print("\n========== LOGISTICS PERFORMANCE KPIs ==========")

print(f"Total orders: {total_orders:,}")
print(f"Delivered orders: {total_delivered:,}")
print(f"Deliveries with valid dates: {valid_delivery_count:,}")

print(f"\nOn-time deliveries: {on_time_count:,}")
print(f"Late deliveries: {late_count:,}")
print(f"On-time delivery rate: {on_time_rate:.2f}%")
print(f"Late delivery rate: {late_rate:.2f}%")

print(f"\nAverage delivery duration: {avg_delivery_days:.2f} days")
print(f"Median delivery duration: {median_delivery_days:.2f} days")

print(f"\nOrders with item data: {len(orders_with_items):,}")
print(f"Average freight per order: {avg_freight:.2f}")
print(f"Total freight charges: {total_freight:,.2f}")
print(f"Average product value per order: {avg_product_value:.2f}")
print(f"Average order value including freight: {avg_order_value:.2f}")

# 6. Descriptive statistics by metric group

print("\n========== DELIVERY DURATION STATISTICS ==========")
print(
    delivered["delivery_duration_days"]
    .dropna()
    .describe()
    .round(2)
)

print("\n========== DELIVERY DELAY STATISTICS ==========")
print(
    valid_deliveries["delivery_delay_days"]
    .describe()
    .round(2)
)

print("\n========== FREIGHT AND ORDER STATISTICS ==========")
print(
    orders_with_items[
        ["total_freight", "total_product_price", "item_count"]
    ]
    .describe()
    .round(2)
)

print("\nKPI analysis completed!")
