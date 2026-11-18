
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(
    "data/processed/week4_modeling_dataset.csv"
)

df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"]
)

df = df.sort_values(
    "order_purchase_timestamp"
).reset_index(drop=True)

# Features and target
numeric_features = [
    "item_count",
    "total_product_price",
    "total_freight",
    "purchase_year",
    "purchase_month_num",
    "purchase_dayofweek",
    "purchase_hour"
]

categorical_features = ["customer_state"]

feature_columns = numeric_features + categorical_features
target_column = "delivery_duration_days"

X = df[feature_columns]
y = df[target_column]

# Same chronological split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = df["order_purchase_timestamp"].iloc[
    split_index:
]

# Load trained Random Forest
model = joblib.load(
    "models/week4_random_forest.pkl"
)

# Predict
predictions = model.predict(X_test)

# Overall evaluation
print("\n========== RANDOM FOREST EVALUATION ==========")

print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
print(
    f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.4f}"
)
print(f"R2: {r2_score(y_test, predictions):.4f}")

# Monthly test performance
results = pd.DataFrame({
    "month": test_dates.dt.to_period("M").astype(str).values,
    "actual": y_test.values,
    "predicted": predictions
})

results["absolute_error"] = (
    results["actual"] - results["predicted"]
).abs()

monthly_results = results.groupby("month").agg(
    orders=("actual", "count"),
    actual_avg_days=("actual", "mean"),
    predicted_avg_days=("predicted", "mean"),
    mae=("absolute_error", "mean")
).reset_index()

print("\n========== MONTHLY TEST PERFORMANCE ==========")
print(monthly_results.round(2).to_string(index=False))

# Save results
monthly_results.to_csv(
    "reports/week4_monthly_model_performance.csv",
    index=False
)

print("\nMonthly evaluation saved successfully!")
