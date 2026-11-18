
from pathlib import Path
import pandas as pd


# 1. Define required files
required_files = [
    "data/processed/week4_modeling_dataset.csv",

    "models/week4_linear_regression.pkl",
    "models/week4_decision_tree.pkl",
    "models/week4_random_forest.pkl",
    "models/week4_random_forest_tuned.pkl",

    "reports/week4_feature_importance.csv",
    "reports/week4_monthly_model_performance.csv",
    "reports/week4_random_forest_tuning_results.csv",
    "reports/week4_model_comparison_after_tuning.csv",
    "reports/week4_tuned_model_predictions.csv",
    "reports/week4_prediction_error_details.csv",
    "reports/week4_prediction_error_by_month.csv",
    "reports/week4_regional_prediction_errors.csv",
    "reports/week4_optimization_recommendations.csv"
]


print("\n========== WEEK 4 FINAL VALIDATION ==========")


# 2. Check file existence
missing_files = []

for file_path in required_files:
    path = Path(file_path)

    if path.exists():
        print(f"[OK] {file_path}")
    else:
        print(f"[MISSING] {file_path}")
        missing_files.append(file_path)


# 3. Validate modeling dataset
dataset = pd.read_csv(
    "data/processed/week4_modeling_dataset.csv"
)

print("\n========== MODELING DATASET ==========")
print("Rows:", len(dataset))
print("Columns:", len(dataset.columns))
print("Missing values:", dataset.isnull().sum().sum())


# 4. Validate model comparison
comparison = pd.read_csv(
    "reports/week4_model_comparison_after_tuning.csv"
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))


# 5. Validate prediction records
predictions = pd.read_csv(
    "reports/week4_tuned_model_predictions.csv"
)

print("\n========== PREDICTION VALIDATION ==========")
print("Prediction records:", len(predictions))
print(
    "Missing prediction values:",
    predictions.isnull().sum().sum()
)


# 6. Validate regional analysis
regional = pd.read_csv(
    "reports/week4_regional_prediction_errors.csv"
)

print("\n========== REGIONAL VALIDATION ==========")
print("Number of regions:", len(regional))
print("Total regional orders:", regional["orders"].sum())


# 7. Validate optimization recommendations
recommendations = pd.read_csv(
    "reports/week4_optimization_recommendations.csv"
)

print("\n========== RECOMMENDATION VALIDATION ==========")
print("Recommendations:", len(recommendations))


# 8. Final status
print("\n========== FINAL STATUS ==========")

if (
    not missing_files
    and dataset.isnull().sum().sum() == 0
    and len(predictions) == 19296
    and predictions.isnull().sum().sum() == 0
    and regional["orders"].sum() == len(predictions)
    and len(recommendations) > 0
):
    print("WEEK 4 VALIDATION PASSED!")

else:
    print("VALIDATION NEEDS ATTENTION.")

    if missing_files:
        print("Missing files:", missing_files)
