
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


# 1. Load modeling dataset
df = pd.read_csv(
    "data/processed/week4_modeling_dataset.csv"
)

# 2. Sort chronologically
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"]
)

df = df.sort_values(
    "order_purchase_timestamp"
).reset_index(drop=True)


# 3. Define features and target
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


# 4. Chronological 80:20 split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

print("\n========== HYPERPARAMETER TUNING ==========")
print(f"Training records: {len(X_train)}")
print("Final test set remains untouched.")


# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            ["customer_state"]
        )
    ],
    remainder="passthrough"
)


# 6. Create Random Forest pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# 7. Define a small hyperparameter grid
param_grid = {
    "model__n_estimators": [100],
    "model__max_depth": [10, 15],
    "model__min_samples_leaf": [10, 20]
}


# 8. Time-aware cross-validation
time_cv = TimeSeriesSplit(n_splits=3)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=time_cv,
    n_jobs=1,
    verbose=2,
    return_train_score=True
)


# 9. Run tuning on training data only
grid_search.fit(X_train, y_train)


# 10. Display best results
print("\n========== BEST MODEL RESULTS ==========")

print("Best parameters:")
print(grid_search.best_params_)

print(
    f"\nBest cross-validation MAE: "
    f"{-grid_search.best_score_:.4f} days"
)


# 11. Save tuning results
results = pd.DataFrame(grid_search.cv_results_)

results["mean_test_MAE"] = (
    -results["mean_test_score"]
)

results["mean_train_MAE"] = (
    -results["mean_train_score"]
)

results = results.sort_values(
    "rank_test_score"
)

results.to_csv(
    "reports/week4_random_forest_tuning_results.csv",
    index=False
)


# 12. Save best trained model
joblib.dump(
    grid_search.best_estimator_,
    "models/week4_random_forest_tuned.pkl"
)

print("\nTuning results saved successfully!")
print("Best model saved successfully!")
