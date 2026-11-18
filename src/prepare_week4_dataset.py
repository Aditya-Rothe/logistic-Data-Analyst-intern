
import pandas as pd

# Load Week 3 dataset
df = pd.read_csv(
    "data/processed/olist_order_level_eda.csv"
)

print("\n========== WEEK 4 DATA PREPARATION ==========")

# Convert purchase timestamp to datetime
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"],
    errors="coerce"
)

# Keep only orders with a valid delivery target
df = df.dropna(subset=[
    "delivery_duration_days",
    "order_purchase_timestamp"
]).copy()

print("\nRecords after removing missing targets:")
print(len(df))

# Extract features from purchase timestamp
df["purchase_year"] = (
    df["order_purchase_timestamp"].dt.year
)

df["purchase_month_num"] = (
    df["order_purchase_timestamp"].dt.month
)

df["purchase_dayofweek"] = (
    df["order_purchase_timestamp"].dt.day_of_week
)

df["purchase_hour"] = (
    df["order_purchase_timestamp"].dt.hour
)

# Select model features
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

# Keep only required columns
model_df = df[
    ["order_purchase_timestamp"] + feature_columns + [target_column]
].copy()

print("\nMissing values before feature cleaning:")
print(model_df.isnull().sum())

# Remove records with missing predictor values
initial_rows = len(model_df)

model_df = model_df.dropna(
    subset=feature_columns
).copy()

removed_rows = initial_rows - len(model_df)

print("\nRecords removed due to missing features:")
print(removed_rows)

# Check final dataset
print("\nFinal modeling dataset shape:")
print(model_df.shape)

print("\nFinal missing values:")
print(model_df.isnull().sum())

print("\nTarget variable summary:")
print(model_df[target_column].describe())

# Save prepared dataset
output_path = "data/processed/week4_modeling_dataset.csv"

model_df.to_csv(output_path, index=False)

print(f"\nPrepared dataset saved to: {output_path}")

print("\nWeek 4 dataset preparation completed!")
