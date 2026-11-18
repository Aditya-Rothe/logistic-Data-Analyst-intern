
import pandas as pd
import joblib

# Load trained Random Forest pipeline
model = joblib.load(
    "models/week4_random_forest.pkl"
)

# Extract preprocessing and trained model
preprocessor = model.named_steps["preprocessor"]
random_forest = model.named_steps["model"]

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Get feature importance scores
importance_values = random_forest.feature_importances_

# Create importance dataframe
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)

# Display top 15 features
print("\n========== RANDOM FOREST FEATURE IMPORTANCE ==========")

print(
    importance_df.head(15).round(4).to_string(index=False)
)

# Save results
importance_df.to_csv(
    "reports/week4_feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
