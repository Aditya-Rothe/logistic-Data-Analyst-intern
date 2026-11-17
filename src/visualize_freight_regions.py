
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("visualizations")

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 2. LOAD DATASETS
# ==========================================

items = pd.read_csv(
    DATA_DIR / "olist_order_items_dataset.csv"
)

products = pd.read_csv(
    DATA_DIR / "olist_products_dataset.csv"
)

orders = pd.read_csv(
    DATA_DIR / "olist_order_level_eda.csv"
)

# ==========================================
# 3. CHART 5: FREIGHT VS PRODUCT WEIGHT
# ==========================================

item_analysis = items.merge(
    products[["product_id", "product_weight_g"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

valid_items = item_analysis.dropna(
    subset=["product_weight_g", "freight_value"]
).copy()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=valid_items,
    x="product_weight_g",
    y="freight_value",
    alpha=0.35,
    s=25
)

plt.title(
    "Relationship Between Product Weight and Freight Charges",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Product Weight (grams)")
plt.ylabel("Freight Charges (BRL)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_freight_vs_product_weight.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 4. CHART 6: REGIONAL LATE DELIVERY RATE
# ==========================================

orders["is_delivered"] = (
    orders["order_status"] == "delivered"
)

orders["valid_delivery"] = (
    orders["is_delivered"]
    & orders["delivery_delay_days"].notna()
)

orders["late"] = (
    orders["valid_delivery"]
    & (orders["delivery_delay_days"] > 0)
)

regional_summary = (
    orders.groupby("customer_state")
    .agg(
        valid_deliveries=("valid_delivery", "sum"),
        late_orders=("late", "sum")
    )
)

regional_summary["late_rate"] = (
    regional_summary["late_orders"]
    / regional_summary["valid_deliveries"] * 100
)

# Include only states with at least 100 valid deliveries
reliable_states = regional_summary[
    regional_summary["valid_deliveries"] >= 100
].copy()

# Select the 10 states with highest late delivery rates
top_late_states = (
    reliable_states
    .sort_values("late_rate", ascending=False)
    .head(10)
    .sort_values("late_rate", ascending=True)
    .reset_index()
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_late_states,
    x="late_rate",
    y="customer_state"
)

plt.title(
    "Top 10 States by Late Delivery Rate",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Late Delivery Rate (%)")
plt.ylabel("Customer State")

# Display percentage labels
for index, value in enumerate(top_late_states["late_rate"]):
    plt.text(
        value + 0.2,
        index,
        f"{value:.2f}%",
        va="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_regional_late_delivery_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 5. FINAL SUMMARY
# ==========================================

print("\n========== VISUALIZATION SUMMARY ==========")

print("Charts generated: 2")

print(
    "- visualizations/05_freight_vs_product_weight.png"
)

print(
    "- visualizations/06_regional_late_delivery_rate.png"
)

print("\nTop 10 states by late delivery rate:")
print(
    top_late_states[
        ["customer_state", "valid_deliveries", "late_rate"]
    ].to_string(index=False)
)

print("\nAll additional visualizations saved successfully!")
