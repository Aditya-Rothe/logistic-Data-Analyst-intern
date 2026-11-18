
import pandas as pd


# 1. Load regional error analysis
df = pd.read_csv(
    "reports/week4_regional_prediction_errors.csv"
)

# 2. Focus on states with at least 100 test orders
reliable_regions = df[
    df["orders"] >= 100
].copy()

# 3. Identify high-error regions
high_error_regions = reliable_regions[
    reliable_regions["mae"] >= 6
]

# 4. Identify regions with high extreme-error percentages
high_extreme_error_regions = reliable_regions[
    reliable_regions["above_10_days_pct"] >= 15
]

print("\n========== LOGISTICS OPTIMIZATION ANALYSIS ==========")

print("\nHigh prediction-error regions:")

print(
    high_error_regions[
        [
            "customer_state",
            "orders",
            "mae",
            "above_10_days_pct"
        ]
    ].round(2).to_string(index=False)
)

print("\nRegions with at least 15% errors above 10 days:")

print(
    high_extreme_error_regions[
        [
            "customer_state",
            "orders",
            "mae",
            "above_10_days_pct"
        ]
    ].round(2).to_string(index=False)
)


# 5. Prepare recommendations
recommendations = pd.DataFrame([
    {
        "area": "Regional Monitoring",
        "finding": "Higher prediction errors in selected states",
        "recommendation": "Prioritize monitoring in PA, PE, BA and PB",
        "expected_benefit": "Earlier identification of potentially delayed deliveries"
    },
    {
        "area": "Exception Management",
        "finding": "Some orders have prediction errors above 10 days",
        "recommendation": "Investigate unusually delayed orders and establish an exception alert process",
        "expected_benefit": "Improved visibility of delivery disruptions"
    },
    {
        "area": "Model Monitoring",
        "finding": "Prediction performance changes across months",
        "recommendation": "Monitor monthly MAE and retrain when performance deteriorates",
        "expected_benefit": "Better adaptation to changing delivery patterns"
    },
    {
        "area": "Data Collection",
        "finding": "Current dataset lacks route and operational information",
        "recommendation": "Collect GPS, dispatch, distance, carrier, traffic and warehouse data",
        "expected_benefit": "More detailed future logistics analysis"
    },
    {
        "area": "Capacity Planning",
        "finding": "Historical order and delivery patterns vary over time",
        "recommendation": "Use order-volume trends to support staffing and workload planning",
        "expected_benefit": "Better-informed resource allocation"
    }
])

# 6. Save recommendations
recommendations.to_csv(
    "reports/week4_optimization_recommendations.csv",
    index=False
)

print("\n========== OPTIMIZATION RECOMMENDATIONS ==========")

print(
    recommendations.to_string(index=False)
)

print("\nOptimization analysis saved successfully!")
