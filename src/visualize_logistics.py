
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
DATA_PATH = Path("data/processed/olist_order_level_eda.csv")
OUTPUT_DIR = Path("visualizations")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Convert date column
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"], errors="coerce"
)

# Set a consistent chart style
sns.set_theme(style="whitegrid")


# ==========================================
# CHART 1: DELIVERY PERFORMANCE
# ==========================================

performance_order = [
    "on time",
    "late",
    "not delivered",
    "date unavailable"
]

performance_counts = (
    df["delivery_performance"]
    .value_counts()
    .reindex(performance_order, fill_value=0)
)

plt.figure(figsize=(9, 5))

sns.barplot(
    x=performance_counts.index,
    y=performance_counts.values
)

plt.title("Order Delivery Performance", fontsize=14)
plt.xlabel("Delivery Category")
plt.ylabel("Number of Orders")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "01_delivery_performance.png",
    dpi=300
)
plt.close()


# ==========================================
# CHART 2: DELIVERY DURATION DISTRIBUTION
# ==========================================

delivered = df[
    (df["order_status"] == "delivered")
    & (df["delivery_duration_days"].notna())
]

plt.figure(figsize=(9, 5))

sns.histplot(
    data=delivered,
    x="delivery_duration_days",
    bins=50,
    kde=True
)

plt.title("Distribution of Delivery Duration", fontsize=14)
plt.xlabel("Delivery Duration (Days)")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "02_delivery_duration.png",
    dpi=300
)
plt.close()


# ==========================================
# CHART 3: MONTHLY ORDER VOLUME
# ==========================================

monthly_orders = (
    df.dropna(subset=["order_purchase_timestamp"])
    .assign(
        purchase_month=lambda x:
        x["order_purchase_timestamp"].dt.to_period("M")
    )
    .groupby("purchase_month")
    .size()
    .reset_index(name="order_count")
)

monthly_orders["purchase_month"] = (
    monthly_orders["purchase_month"].astype(str)
)

plt.figure(figsize=(12, 5))

sns.lineplot(
    data=monthly_orders,
    x="purchase_month",
    y="order_count",
    marker="o"
)

plt.title("Monthly Order Volume", fontsize=14)
plt.xlabel("Purchase Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "03_monthly_order_volume.png",
    dpi=300
)
plt.close()


# ==========================================
# CHART 4: DELIVERY DELAY DISTRIBUTION
# ==========================================

valid_deliveries = delivered[
    delivered["delivery_delay_days"].notna()
]

plt.figure(figsize=(9, 5))

sns.histplot(
    data=valid_deliveries,
    x="delivery_delay_days",
    bins=50,
    kde=True
)

plt.axvline(
    x=0,
    linestyle="--",
    label="Estimated Delivery Date"
)

plt.title("Distribution of Delivery Delay", fontsize=14)
plt.xlabel("Delivery Delay (Days)")
plt.ylabel("Number of Orders")
plt.legend()

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "04_delivery_delay_distribution.png",
    dpi=300
)
plt.close()


print("\n========== VISUALIZATION SUMMARY ==========")
print("Charts generated:", 4)

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print("-", file)

print("\nAll visualizations saved successfully!")
