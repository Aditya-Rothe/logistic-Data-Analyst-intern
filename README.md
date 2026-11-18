# Logistics Data Analysis & Predictive Modeling

A four-week logistics data analytics project completed as part of the **Logistics Data Analyst Internship at Yuva Intern**.

The project uses the **Brazilian E-Commerce Public Dataset by Olist** to explore delivery performance, identify logistics patterns, analyze regional differences, and build a machine learning model to predict delivery duration.

## Project Objectives

- Analyze e-commerce order and delivery data.
- Identify important logistics performance indicators (KPIs).
- Explore delivery delays, regional performance, and customer satisfaction.
- Build and evaluate machine learning models for delivery duration prediction.
- Propose data-driven recommendations for logistics monitoring and planning.

## Dataset

**Source:** [Brazilian E-Commerce Public Dataset by Olist – Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The dataset contains information about orders, customers, sellers, products, payments, reviews, and delivery performance.

Note: The raw and processed datasets are excluded from this repository. Download the dataset separately from Kaggle.

## Project Timeline

### Week 1 – Strategic Planning and Data Exploration
- Understood the logistics business scenario and project objectives.
- Explored the available Olist datasets and their structure.
- Identified key logistics KPIs and analytical objectives.
- Developed the project roadmap and folder structure.

### Week 2 – Data Cleaning and Preparation
- Inspected data quality and missing values.
- Investigated order statuses and missing delivery information.
- Prepared datasets for further analysis.
- Performed data validation and checked processed outputs.

### Week 3 – Exploratory Data Analysis and Visualization
- Analyzed delivery performance and delivery duration.
- Examined monthly order trends and delivery delay distribution.
- Explored freight charges and product weight.
- Compared regional delivery performance.
- Investigated the relationship between delivery performance and customer satisfaction.
- Generated visualizations to communicate findings.

### Week 4 – Predictive Modeling and Optimization
- Prepared a modeling dataset containing 96,476 valid orders.
- Selected eight features available at purchase time.
- Used a chronological 80:20 train-test split.
- Trained and compared Linear Regression, Decision Tree, and Random Forest models.
- Performed time-series cross-validation for Random Forest hyperparameter tuning.
- Analyzed prediction errors, feature importance, and regional performance.
- Proposed recommendations for regional monitoring, exception management, model monitoring, and capacity planning.

## Week 4 – Model Performance

The original Random Forest model achieved the best test-set results among the evaluated models.

| Model | MAE (Days) | RMSE (Days) | R² |
|---|---:|---:|---:|
| Mean Baseline | 6.4328 | 7.5769 | -0.6492 |
| Linear Regression | 6.6029 | 7.7959 | -0.7459 |
| Decision Tree | 4.6326 | 6.0481 | -0.0508 |
| Random Forest | **4.5071** | **5.9027** | **-0.0009** |
| Tuned Random Forest | 4.6047 | 5.9991 | -0.0339 |

**Key findings:**
- The original Random Forest achieved an MAE of approximately 4.51 days.
- Around 63.19% of test predictions had an absolute error within five days.
- Customer state, freight charges, and purchase month were among the most influential model features.
- Prediction performance varied across regions and months.
- Hyperparameter tuning did not improve the original Random Forest's test performance.

**Important:** The negative R² indicates that the model still has limitations in explaining variation in the test period. It is an experimental model, not a production-ready delivery prediction system.

## Optimization Recommendations

Based on the analysis, the project proposes:

- Prioritizing regional monitoring in areas with higher prediction errors.
- Investigating unusually delayed deliveries through exception alerts.
- Monitoring model performance over time and retraining when necessary.
- Collecting additional operational data such as GPS, distance, traffic, carrier, and warehouse information.
- Using order-volume trends to support workload and capacity planning.

These are proposed strategies, not measured operational improvements.

## Project Structure

```text
logistic-Data-Analyst-intern/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   └── Python analysis and modeling scripts
│
├── notebooks/
├── models/
├── reports/
├── visualizations/
├── docs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Development Environment:** VS Code
- **Version Control:** Git and GitHub

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/Aditya-Rothe/logistic-Data-Analyst-intern.git
cd logistic-Data-Analyst-intern
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the Olist dataset from Kaggle and place the required CSV files in `data/raw/`.

5. Run the relevant Python scripts from the project root using:

```bash
python src/script_name.py
```

Refer to the scripts in the `src/` directory for the individual analysis and modeling steps.

## Author

**Aditya Rothe**

B.Sc. Data Science

[GitHub Profile](https://github.com/Aditya-Rothe)

---

*This project was developed as part of a four-week Logistics Data Analyst Internship assignment, applying practical data analysis, visualization, machine learning, and problem-solving techniques to an e-commerce logistics dataset.*