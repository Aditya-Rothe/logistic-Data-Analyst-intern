
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/olist_order_level_eda.csv")

df = pd.read_csv(DATA_PATH)

# Convert dates
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"], errors="coerce"
)

print("\n========== CHART 1: DELIVERY PERFORMANCE ==========")

performance = df["delivery_performance"].value_counts()

print(performance)
print("\nPercentage distribution:")
print((performance / len(df) * 100).round(2))


print("\n========== CHART 2: DELIVERY DURATION ==========")

delivered = df[
    (df["order_status"] == "delivered")
    & df["delivery_duration_days"].notna()
]

print("Average:", round(delivered["delivery_duration_days"].mean(), 2))
print("Median:", round(delivered["delivery_duration_days"].median(), 2))
print("25th percentile:", round(delivered["delivery_duration_days"].quantile(0.25), 2))
print("75th percentile:", round(delivered["delivery_duration_days"].quantile(0.75), 2))
print("90th percentile:", round(delivered["delivery_duration_days"].quantile(0.90), 2))
print("Maximum:", round(delivered["delivery_duration_days"].max(), 2))


print("\n========== CHART 3: MONTHLY ORDER VOLUME ==========")

monthly_orders = (
    df.dropna(subset=["order_purchase_timestamp"])
    .groupby(
        df["order_purchase_timestamp"].dt.to_period("M")
    )
    .size()
    .sort_index()
)

print(monthly_orders)

print("\nHighest order volume:")
print(monthly_orders.idxmax(), "-", monthly_orders.max(), "orders")

print("\nLowest order volume:")
print(monthly_orders.idxmin(), "-", monthly_orders.min(), "orders")


print("\n========== CHART 4: DELIVERY DELAY ==========")

valid_deliveries = df[
    (df["order_status"] == "delivered")
    & df["delivery_delay_days"].notna()
]

early = (valid_deliveries["delivery_delay_days"] < 0).sum()
exact = (valid_deliveries["delivery_delay_days"] == 0).sum()
late = (valid_deliveries["delivery_delay_days"] > 0).sum()

print("Delivered early:", early)
print("Delivered exactly on estimated date:", exact)
print("Delivered late:", late)

print("\nPercentage distribution:")
print("Early:", round(early / len(valid_deliveries) * 100, 2), "%")
print("Exactly on date:", round(exact / len(valid_deliveries) * 100, 2), "%")
print("Late:", round(late / len(valid_deliveries) * 100, 2), "%")


# ==========================================
# ADDITIONAL CHECK: CALENDAR-DATE DELIVERY
# ==========================================

actual = pd.to_datetime(
    df["order_delivered_customer_date"],
    errors="coerce"
)

estimated = pd.to_datetime(
    df["order_estimated_delivery_date"],
    errors="coerce"
)

delivered_mask = df["order_status"].eq("delivered")

calendar_delay = (
    actual.dt.normalize() - estimated.dt.normalize()
).dt.days

valid_calendar_delay = calendar_delay[delivered_mask].dropna()

print("\n========== CALENDAR-DATE DELIVERY CHECK ==========")

print("Purchase date range:")
print(
    df["order_purchase_timestamp"].min(),
    "to",
    df["order_purchase_timestamp"].max()
)

print("\nEstimated delivery timestamp hours:")
print(estimated.dt.hour.value_counts().sort_index())

print("\nCalendar-date delivery performance:")
print("Early:", (valid_calendar_delay < 0).sum())
print("Exactly on date:", (valid_calendar_delay == 0).sum())
print("Late:", (valid_calendar_delay > 0).sum())

print("\nCalendar-date on-time rate:")
print(
    (valid_calendar_delay <= 0).mean() * 100
)

print("\nCalendar-date late rate:")
print(
    (valid_calendar_delay > 0).mean() * 100
)


print("\nVisualization inspection completed!")
