
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("reports")
VISUAL_DIR = Path("visualizations")

OUTPUT_DIR.mkdir(exist_ok=True)
VISUAL_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 2. LOAD DATASETS
# ==========================================

orders = pd.read_csv(
    DATA_DIR / "olist_order_level_eda.csv"
)

reviews = pd.read_csv(
    DATA_DIR / "olist_order_reviews_dataset.csv"
)

print("\n========== DATASET OVERVIEW ==========")

print("Total orders:", len(orders))
print("Total review records:", len(reviews))
print("Unique reviewed orders:", reviews["order_id"].nunique())

# ==========================================
# 3. PREPARE REVIEW DATA
# ==========================================

reviews["review_score"] = pd.to_numeric(
    reviews["review_score"],
    errors="coerce"
)

# Aggregate reviews to one score per order.
# This prevents multiple review records from
# duplicating an order during the merge.

order_reviews = (
    reviews.groupby("order_id")
    .agg(
        review_score=("review_score", "mean"),
        review_records=("review_score", "count")
    )
    .reset_index()
)

# ==========================================
# 4. PREPARE DELIVERY PERFORMANCE
# ==========================================

orders["delivery_performance"] = "not eligible"

valid_delivery = (
    (orders["order_status"] == "delivered")
    & orders["delivery_delay_days"].notna()
)

orders.loc[
    valid_delivery
    & (orders["delivery_delay_days"] <= 0),
    "delivery_performance"
] = "on time"

orders.loc[
    valid_delivery
    & (orders["delivery_delay_days"] > 0),
    "delivery_performance"
] = "late"

# Keep only delivered orders with valid delivery dates
eligible_orders = orders[
    orders["delivery_performance"].isin(["on time", "late"])
].copy()

# ==========================================
# 5. MERGE ORDERS AND CUSTOMER REVIEWS
# ==========================================

analysis_df = eligible_orders.merge(
    order_reviews,
    on="order_id",
    how="left",
    validate="one_to_one"
)

# Keep orders that have valid review scores
review_analysis = analysis_df.dropna(
    subset=["review_score"]
).copy()

print("\n========== REVIEW DATA VALIDATION ==========")

print("Eligible delivered orders:", len(eligible_orders))
print("Orders with available review scores:", len(review_analysis))
print(
    "Orders without review scores:",
    len(eligible_orders) - len(review_analysis)
)

print(
    "Duplicate order IDs after merge:",
    review_analysis["order_id"].duplicated().sum()
)

# ==========================================
# 6. CUSTOMER SATISFACTION ANALYSIS
# ==========================================

satisfaction_summary = (
    review_analysis.groupby("delivery_performance")
    .agg(
        reviewed_orders=("order_id", "count"),
        average_review_score=("review_score", "mean"),
        median_review_score=("review_score", "median")
    )
    .round(2)
)

print("\n========== CUSTOMER SATISFACTION ANALYSIS ==========")
print(satisfaction_summary)

# ==========================================
# 7. REVIEW SCORE DISTRIBUTION
# ==========================================

score_distribution = pd.crosstab(
    review_analysis["delivery_performance"],
    review_analysis["review_score"]
)

score_percentage = (
    pd.crosstab(
        review_analysis["delivery_performance"],
        review_analysis["review_score"],
        normalize="index"
    ) * 100
).round(2)

print("\n========== REVIEW SCORE COUNTS ==========")
print(score_distribution)

print("\n========== REVIEW SCORE DISTRIBUTION (%) ==========")
print(score_percentage)

# ==========================================
# 8. SAVE ANALYSIS RESULTS
# ==========================================

satisfaction_summary.to_csv(
    OUTPUT_DIR / "customer_satisfaction_summary.csv"
)

score_distribution.to_csv(
    OUTPUT_DIR / "review_score_distribution_counts.csv"
)

score_percentage.to_csv(
    OUTPUT_DIR / "review_score_distribution_percentages.csv"
)

# ==========================================
# 9. CHART 7: DELIVERY PERFORMANCE VS REVIEW
# ==========================================

score_long = (
    score_percentage
    .rename_axis("delivery_performance")
    .reset_index()
    .melt(
        id_vars="delivery_performance",
        var_name="review_score",
        value_name="percentage"
    )
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=score_long,
    x="review_score",
    y="percentage",
    hue="delivery_performance"
)

plt.title(
    "Customer Review Score Distribution by Delivery Performance",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Customer Review Score")
plt.ylabel("Percentage of Reviewed Orders (%)")

plt.legend(title="Delivery Performance")

plt.tight_layout()

plt.savefig(
    VISUAL_DIR / "07_delivery_vs_customer_satisfaction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========")

print("Analysis records:", len(review_analysis))
print("Delivery categories:", review_analysis[
    "delivery_performance"
].nunique())

print(
    "\nChart generated:"
    "\n- visualizations/07_delivery_vs_customer_satisfaction.png"
)

print("\nAnalysis results saved in reports/")
print("\nCustomer satisfaction analysis completed!")
