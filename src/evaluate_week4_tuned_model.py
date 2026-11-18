
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


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

X = df[features]
y = df[target]


# 3. Recreate chronological split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_timestamps = df["order_purchase_timestamp"].iloc[
    split_index:
]


# 4. Load original and tuned models
original_model = joblib.load(
    "models/week4_random_forest.pkl"
)

tuned_model = joblib.load(
    "models/week4_random_forest_tuned.pkl"
)


# 5. Generate predictions
baseline_predictions = [y_train.mean()] * len(y_test)

original_predictions = original_model.predict(X_test)

tuned_predictions = tuned_model.predict(X_test)


# 6. Evaluation function
def evaluate_model(name, actual, predicted):

    return {
        "Model": name,
        "MAE": mean_absolute_error(
            actual, predicted
        ),
        "RMSE": mean_squared_error(
            actual, predicted
        ) ** 0.5,
        "R2": r2_score(
            actual, predicted
        )
    }


# 7. Compare all models
results = []

results.append(
    evaluate_model(
        "Baseline",
        y_test,
        baseline_predictions
    )
)

results.append(
    evaluate_model(
        "Original Random Forest",
        y_test,
        original_predictions
    )
)

results.append(
    evaluate_model(
        "Tuned Random Forest",
        y_test,
        tuned_predictions
    )
)


comparison = pd.DataFrame(results)

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    comparison.round(4).to_string(index=False)
)


# 8. Save comparison
comparison.to_csv(
    "reports/week4_model_comparison_after_tuning.csv",
    index=False
)


# 9. Save tuned model predictions for further analysis
prediction_df = pd.DataFrame({
    "order_purchase_timestamp": test_timestamps.values,
    "actual_delivery_days": y_test.values,
    "original_rf_prediction": original_predictions,
    "tuned_rf_prediction": tuned_predictions
})

prediction_df.to_csv(
    "reports/week4_tuned_model_predictions.csv",
    index=False
)


print("\nModel comparison saved successfully!")
print("Tuned model predictions saved successfully!")
