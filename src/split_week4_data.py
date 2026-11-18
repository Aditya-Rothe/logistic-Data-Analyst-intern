
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load prepared dataset
df = pd.read_csv(
    "data/processed/week4_modeling_dataset.csv"
)

print("\n========== WEEK 4 TRAIN-TEST SPLIT ==========")

# Convert timestamp to datetime
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"]
)

# Sort orders chronologically
df = df.sort_values(
    "order_purchase_timestamp"
).reset_index(drop=True)

# Define input features and target
feature_columns = [
    "item_count",
    "total_product_price",
    "total_freight",
    "customer_state",
    "purchase_year",
    "purchase_month_num",
    "purchase_dayofweek",
    "purchase_hour"
]

target_column = "delivery_duration_days"

X = df[feature_columns]
y = df[target_column]

# Chronological 80:20 split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining date range:")
print(
    df["order_purchase_timestamp"].iloc[0],
    "to",
    df["order_purchase_timestamp"].iloc[split_index - 1]
)

print("\nTesting date range:")
print(
    df["order_purchase_timestamp"].iloc[split_index],
    "to",
    df["order_purchase_timestamp"].iloc[-1]
)

# Baseline model: predict training target mean for every test order
baseline_prediction = np.full(
    shape=len(y_test),
    fill_value=y_train.mean()
)

# Evaluate baseline
mae = mean_absolute_error(
    y_test, baseline_prediction
)

rmse = np.sqrt(
    mean_squared_error(y_test, baseline_prediction)
)

r2 = r2_score(
    y_test, baseline_prediction
)

print("\n========== BASELINE MODEL RESULTS ==========")
print(f"Training target mean: {y_train.mean():.2f} days")
print(f"Testing target mean: {y_test.mean():.2f} days")
print(f"MAE: {mae:.4f} days")
print(f"RMSE: {rmse:.4f} days")
print(f"R2 Score: {r2:.4f}")

print("\nTrain-test split completed successfully!")
