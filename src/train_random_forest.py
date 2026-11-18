
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(
    "data/processed/week4_modeling_dataset.csv"
)

# Sort chronologically
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"]
)

df = df.sort_values(
    "order_purchase_timestamp"
).reset_index(drop=True)

# Define features and target
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

# Chronological 80:20 split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# Random Forest pipeline
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# Train model
random_forest_model.fit(X_train, y_train)

# Predictions
y_pred = random_forest_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== RANDOM FOREST RESULTS ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(f"MAE: {mae:.4f} days")
print(f"RMSE: {rmse:.4f} days")
print(f"R2 Score: {r2:.4f}")

# Save model
joblib.dump(
    random_forest_model,
    "models/week4_random_forest.pkl"
)

print("\nRandom Forest model saved successfully!")
