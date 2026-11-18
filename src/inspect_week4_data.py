
import pandas as pd

# Load Week 3 order-level dataset
df = pd.read_csv(
    "data/processed/olist_order_level_eda.csv"
)

print("\n========== WEEK 4 DATA INSPECTION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDelivery Duration Summary:")
print(df["delivery_duration_days"].describe())

print("\nSample Records:")
print(df.head())

print("\nTotal Valid Delivery Records:")
print(df["delivery_duration_days"].notna().sum())

print("\nTotal Missing Target Records:")
print(df["delivery_duration_days"].isna().sum())
