
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error


# 1. Load dataset
df = pd.read_csv(
    "data/processed/week4_modeling_dataset.csv"
)

df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"]
)

df = df.sort_values(
    "order_purchase_timestamp"
).reset_index(drop=True)


# 2. Define features and target
features = [
    "item_count",
    "total_product_price",
    "total_freight",
    "customer_state",
    "purchase_year",
    "purchase_month_num",
    "purchase_dayofweek",
    "purchase_hour"
]

target = "delivery_duration_days"


# 3. Recreate chronological test split
split_index = int(len(df) * 0.80)

test_df = df.iloc[split_index:].copy()

X_test = test_df[features]
y_test = test_df[target]


# 4. Load original Random Forest
model = joblib.load(
    "models/week4_random_forest.pkl"
)


# 5. Generate predictions
predictions = model.predict(X_test)


# 6. Calculate prediction errors
error_df = pd.DataFrame({
    "order_purchase_timestamp":
        test_df["order_purchase_timestamp"].values,

    "customer_state":
        test_df["customer_state"].values,

    "actual_delivery_days":
        y_test.values,

    "predicted_delivery_days":
        predictions
})

error_df["absolute_error"] = abs(
    error_df["actual_delivery_days"]
    - error_df["predicted_delivery_days"]
)

error_df["prediction_error"] = (
    error_df["predicted_delivery_days"]
    - error_df["actual_delivery_days"]
)


# 7. Error distribution
print("\n========== PREDICTION ERROR ANALYSIS ==========")

print("\nAbsolute Error Statistics:")

print(
    error_df["absolute_error"].describe().round(2)
)


# 8. Error ranges
error_df["error_category"] = pd.cut(
    error_df["absolute_error"],
    bins=[-np.inf, 2, 5, 10, np.inf],
    labels=[
        "0-2 days",
        "2-5 days",
        "5-10 days",
        "Above 10 days"
    ]
)

print("\nPrediction Error Categories:")

print(
    error_df["error_category"]
    .value_counts()
    .sort_index()
)


# 9. Largest prediction errors
print("\nTop 10 Largest Prediction Errors:")

print(
    error_df.nlargest(
        10, "absolute_error"
    )[
        [
            "order_purchase_timestamp",
            "customer_state",
            "actual_delivery_days",
            "predicted_delivery_days",
            "absolute_error"
        ]
    ].round(2).to_string(index=False)
)


# 10. Monthly error analysis
error_df["month"] = (
    error_df["order_purchase_timestamp"]
    .dt.to_period("M")
)

monthly_errors = (
    error_df.groupby("month", observed=True)
    .agg(
        orders=("absolute_error", "count"),
        actual_avg_days=("actual_delivery_days", "mean"),
        predicted_avg_days=("predicted_delivery_days", "mean"),
        mae=("absolute_error", "mean")
    )
    .reset_index()
)

print("\nMonthly Prediction Errors:")

print(
    monthly_errors.round(2).to_string(index=False)
)


# 11. Save results
error_df.to_csv(
    "reports/week4_prediction_error_details.csv",
    index=False
)

monthly_errors.to_csv(
    "reports/week4_prediction_error_by_month.csv",
    index=False
)

print("\nPrediction error analysis saved successfully!")
