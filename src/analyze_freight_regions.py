
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

# ==========================================
# 1. LOAD DATASETS
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
# 2. FREIGHT VS PRODUCT WEIGHT
# ==========================================

item_analysis = items.merge(
    products[["product_id", "product_weight_g"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Keep rows with valid weight and freight
valid_items = item_analysis.dropna(
    subset=["product_weight_g", "freight_value"]
).copy()

print("\n========== FREIGHT VS PRODUCT WEIGHT ==========")

print("Total order items:", len(item_analysis))
print("Items with valid weight and freight:", len(valid_items))

# ==========================================
# 2A. MISSING PRODUCT WEIGHT VALIDATION
# ==========================================

missing_weight_items = item_analysis[
    item_analysis["product_weight_g"].isna()
]

print("\n========== MISSING PRODUCT WEIGHT CHECK ==========")

print(
    "Items with missing product weight:",
    len(missing_weight_items)
)

print(
    "Unique products with missing weight:",
    missing_weight_items["product_id"].nunique()
)

print(
    "Missing freight values:",
    item_analysis["freight_value"].isna().sum()
)

print(
    "Items excluded from analysis:",
    len(item_analysis) - len(valid_items)
)

# ==========================================
# 2B. CORRELATION ANALYSIS
# ==========================================

pearson_corr = valid_items[
    ["product_weight_g", "freight_value"]
].corr(method="pearson").iloc[0, 1]

spearman_corr = valid_items[
    ["product_weight_g", "freight_value"]
].corr(method="spearman").iloc[0, 1]

print(f"\nPearson correlation: {pearson_corr:.3f}")
print(f"Spearman correlation: {spearman_corr:.3f}")

# ==========================================
# 2C. FREIGHT BY WEIGHT CATEGORY
# ==========================================

weight_bins = [
    0, 500, 1000, 5000, 10000, float("inf")
]

weight_labels = [
    "0-500 g",
    "500 g-1 kg",
    "1-5 kg",
    "5-10 kg",
    "Above 10 kg"
]

valid_items["weight_category"] = pd.cut(
    valid_items["product_weight_g"],
    bins=weight_bins,
    labels=weight_labels,
    include_lowest=True
)

weight_summary = (
    valid_items.groupby(
        "weight_category",
        observed=False
    )
    .agg(
        item_count=("product_id", "count"),
        average_weight_g=("product_weight_g", "mean"),
        average_freight=("freight_value", "mean")
    )
    .round(2)
)

print("\nFreight by product weight category:")
print(weight_summary)


# ==========================================
# 3. REGIONAL LOGISTICS PERFORMANCE
# ==========================================

# Create analysis flags
orders["is_delivered"] = (
    orders["order_status"] == "delivered"
)

orders["valid_delivery"] = (
    orders["is_delivered"]
    & orders["delivery_delay_days"].notna()
)

orders["on_time"] = (
    orders["valid_delivery"]
    & (orders["delivery_delay_days"] <= 0)
)

orders["late"] = (
    orders["valid_delivery"]
    & (orders["delivery_delay_days"] > 0)
)

# Group by customer state
regional_summary = (
    orders.groupby("customer_state")
    .agg(
        total_orders=("order_id", "count"),
        average_freight=("total_freight", "mean"),
        delivered_orders=("is_delivered", "sum"),
        valid_deliveries=("valid_delivery", "sum"),
        on_time_orders=("on_time", "sum"),
        late_orders=("late", "sum"),
        average_delivery_days=("delivery_duration_days", "mean")
    )
)

# Calculate regional delivery rates
regional_summary["on_time_rate"] = (
    regional_summary["on_time_orders"]
    / regional_summary["valid_deliveries"] * 100
)

regional_summary["late_rate"] = (
    regional_summary["late_orders"]
    / regional_summary["valid_deliveries"] * 100
)

regional_summary = regional_summary.round(2)

print("\n========== REGIONAL LOGISTICS ANALYSIS ==========")

print("\nTop 10 states by order volume:")
print(
    regional_summary.sort_values(
        "total_orders", ascending=False
    ).head(10)
)

# Avoid drawing conclusions from states with very few deliveries
reliable_states = regional_summary[
    regional_summary["valid_deliveries"] >= 100
]

print("\nTop 10 states by late delivery rate")
print("(Only states with at least 100 valid deliveries)")

print(
    reliable_states.sort_values(
        "late_rate", ascending=False
    ).head(10)
)

print("\nTop 10 states by average freight:")
print(
    regional_summary.sort_values(
        "average_freight", ascending=False
    ).head(10)
)

# ==========================================
# 4. SAVE ANALYSIS RESULTS
# ==========================================

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

weight_summary.to_csv(
    OUTPUT_DIR / "freight_by_weight_category.csv"
)

regional_summary.to_csv(
    OUTPUT_DIR / "regional_logistics_summary.csv"
)

print("\nAnalysis summaries saved in reports/")
print("\nFreight and regional analysis completed!")
