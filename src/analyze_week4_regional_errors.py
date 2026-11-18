
import pandas as pd


# 1. Load prediction error details
df = pd.read_csv(
    "reports/week4_prediction_error_details.csv"
)


# 2. Group errors by customer state
regional_analysis = (
    df.groupby("customer_state")
    .agg(
        orders=("absolute_error", "count"),
        actual_avg_days=("actual_delivery_days", "mean"),
        predicted_avg_days=("predicted_delivery_days", "mean"),
        mae=("absolute_error", "mean"),
        median_error=("absolute_error", "median"),
        orders_above_10_days=(
            "absolute_error",
            lambda x: (x > 10).sum()
        )
    )
    .reset_index()
)


# 3. Calculate percentage of large errors
regional_analysis["above_10_days_pct"] = (
    regional_analysis["orders_above_10_days"]
    / regional_analysis["orders"]
) * 100


# 4. Sort by MAE
regional_analysis = regional_analysis.sort_values(
    "mae",
    ascending=False
).reset_index(drop=True)


# 5. Display results
print("\n========== REGIONAL PREDICTION ERROR ANALYSIS ==========")

print(
    regional_analysis.round(2).to_string(index=False)
)


# 6. Identify regions with at least 100 test orders
reliable_regions = regional_analysis[
    regional_analysis["orders"] >= 100
]

print("\n========== REGIONS WITH HIGHEST MAE (100+ ORDERS) ==========")

print(
    reliable_regions.head(10)
    .round(2)
    .to_string(index=False)
)


# 7. Save results
regional_analysis.to_csv(
    "reports/week4_regional_prediction_errors.csv",
    index=False
)

print("\nRegional analysis saved successfully!")
