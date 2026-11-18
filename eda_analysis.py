from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")





# -------------------- 1. LOAD DATA --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retail_sales.csv"
OUTPUT_DIR = BASE_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE.name}. Keep it in the same folder as this script."
    )

df = pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])

print("=" * 65)
print("CODEALPHA INTERNSHIP | TASK 2: EXPLORATORY DATA ANALYSIS")
print("=" * 65)







# -------------------- 2. MEANINGFUL QUESTIONS --------------------
questions = [
    "Which product category generates the highest total sales?",
    "Which month records the highest sales?",
    "Which region contributes the most revenue?",
    "How are quantity, unit price, sales and profit related?",
    "Are there missing values, duplicate rows or potential outliers?",
    "Do the two largest product categories have different average order sales?"
]

print("\n1. BUSINESS QUESTIONS")
for number, question in enumerate(questions, start=1):
    print(f"{number}. {question}")



# -------------------- 3. DATA STRUCTURE --------------------
print("\n2. DATA STRUCTURE")
print("Dataset shape (rows, columns):", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types and non-null counts:")
df.info()
print("\nFirst five records:")
print(df.head().to_string(index=False))
print("\nDescriptive statistics:")
print(df.describe(include="all").to_string())



# -------------------- 4. DATA QUALITY CHECKS --------------------
print("\n3. DATA QUALITY CHECKS")
missing = df.isnull().sum()
duplicates = int(df.duplicated().sum())

print("\nMissing values by column:")
print(missing[missing > 0].to_string() if (missing > 0).any() else "No missing values.")
print("\nDuplicate rows:", duplicates)




# Make a working copy and apply basic, transparent cleaning.
clean = df.copy()
clean["Discount"] = clean["Discount"].fillna(clean["Discount"].median())
clean["Profit"] = clean["Profit"].fillna(clean["Profit"].median())
clean["Region"] = clean["Region"].fillna("Unknown")
clean["Product"] = clean["Product"].fillna("Unknown")
clean = clean.drop_duplicates()




# -------------------- 5. FEATURE ENGINEERING --------------------
clean["Month"] = clean["Order_Date"].dt.to_period("M").astype(str)
clean["Order_Month"] = clean["Order_Date"].dt.month_name()



# -------------------- 6. TRENDS AND PATTERNS --------------------
print("\n4. TRENDS AND PATTERNS")

category_sales = clean.groupby("Category")["Sales"].sum().sort_values(ascending=False)
region_sales = clean.groupby("Region")["Sales"].sum().sort_values(ascending=False)
monthly_sales = clean.groupby("Month")["Sales"].sum().sort_values(ascending=False)

print("\nTotal sales by category:")
print(category_sales.round(2).to_string())
print("\nTotal sales by region:")
print(region_sales.round(2).to_string())
print("\nTop 5 months by sales:")
print(monthly_sales.head(5).round(2).to_string())

if not category_sales.empty:
    print("\nHighest-selling category:", category_sales.index[0])
if not region_sales.empty:
    print("Highest-selling region:", region_sales.index[0])
if not monthly_sales.empty:
    print("Highest-sales month:", monthly_sales.index[0])




# -------------------- 7. HYPOTHESIS TESTING --------------------
print("\n5. HYPOTHESIS TESTING")
print("Question: Do the two largest categories differ in average order sales?")
print("H0: The two categories have equal mean sales.")
print("H1: The two categories have different mean sales.")

top_categories = clean["Category"].value_counts().head(2).index.tolist()
test_result = "Not enough categories to perform the test."
p_value = np.nan

if len(top_categories) == 2:
    sample_a = clean.loc[clean["Category"] == top_categories[0], "Sales"].dropna()
    sample_b = clean.loc[clean["Category"] == top_categories[1], "Sales"].dropna()

    if len(sample_a) >= 2 and len(sample_b) >= 2:
        t_stat, p_value = ttest_ind(sample_a, sample_b, equal_var=False)
        print(f"Group 1: {top_categories[0]} (n={len(sample_a)}, mean={sample_a.mean():.2f})")
        print(f"Group 2: {top_categories[1]} (n={len(sample_b)}, mean={sample_b.mean():.2f})")
        print(f"Welch t-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.6f}")

        if p_value < 0.05:
            test_result = "Reject H0: evidence of a statistically significant difference."
        else:
            test_result = "Fail to reject H0: insufficient evidence of a significant difference."
        print(test_result)





# -------------------- 8. OUTLIER DETECTION --------------------
print("\n6. OUTLIER DETECTION (IQR METHOD)")
outlier_summary = {}
for column in ["Quantity", "Unit_Price", "Sales", "Profit"]:
    q1 = clean[column].quantile(0.25)
    q3 = clean[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((clean[column] < lower) | (clean[column] > upper)).sum())
    outlier_summary[column] = count
    print(f"{column}: {count} potential outlier(s)")





# -------------------- 9. VISUALIZATIONS --------------------
print("\n7. CREATING VISUALIZATIONS...")




# Category sales
plt.figure(figsize=(9, 5))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title("Total Sales by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_sales.png", dpi=150)
plt.close()



# Monthly sales trend
monthly_chrono = clean.groupby("Month")["Sales"].sum().sort_index()
plt.figure(figsize=(11, 5))
plt.plot(monthly_chrono.index, monthly_chrono.values, marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_sales_trend.png", dpi=150)
plt.close()



# Correlation heatmap
numeric_columns = clean.select_dtypes(include=np.number).columns
plt.figure(figsize=(8, 6))
sns.heatmap(clean[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()


# Sales distribution
plt.figure(figsize=(8, 5))
sns.histplot(clean["Sales"], bins=30, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sales_distribution.png", dpi=150)
plt.close()


# Sales outlier boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=clean["Sales"])
plt.title("Sales Outlier Detection")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sales_outliers.png", dpi=150)
plt.close()


# -------------------- 10. KEY FINDINGS AND REPORT --------------------
total_sales = clean["Sales"].sum()
total_profit = clean["Profit"].sum()
average_sales = clean["Sales"].mean()
total_orders = len(clean)

findings = [
    "EDA SUMMARY REPORT",
    "=" * 50,
    f"Rows in original dataset: {len(df)}",
    f"Columns in original dataset: {df.shape[1]}",
    f"Duplicate rows detected: {duplicates}",
    f"Total missing cells detected: {int(missing.sum())}",
    f"Rows after removing duplicates: {len(clean)}",
    f"Total sales after basic cleaning: {total_sales:,.2f}",
    f"Total profit after basic cleaning: {total_profit:,.2f}",
    f"Average order sales: {average_sales:,.2f}",
    f"Number of records after cleaning: {total_orders}",
    "",
    "TOP CATEGORY SALES:",
    category_sales.head(5).round(2).to_string(),
    "",
    "REGIONAL SALES:",
    region_sales.round(2).to_string(),
    "",
    "HYPOTHESIS TEST:",
    test_result,
    f"P-value: {p_value:.6f}" if not np.isnan(p_value) else "P-value: not calculated",
    "",
    "POTENTIAL OUTLIERS (IQR):",
]
findings.extend([f"{key}: {value}" for key, value in outlier_summary.items()])
findings += [
    "",
    "DATA ISSUES TO REVIEW:",
    "- Missing values were found and handled with simple documented replacements.",
    "- Duplicate rows were detected and removed from the working dataset.",
    "- IQR flags potential outliers; these should be investigated, not automatically deleted.",
    "- This is a synthetic practice dataset, not real company data.",
    "",
    "CONCLUSION:",
    "The analysis explored sales patterns across categories, regions and months,",
    "checked data quality, identified potential outliers and performed a statistical test.",
    "Review the generated charts and business context before making decisions."
]

report_text = "\n".join(findings)
(OUTPUT_DIR / "EDA_Report.txt").write_text(report_text, encoding="utf-8")

print("\n" + report_text)
print(f"\nCharts and report saved in: {OUTPUT_DIR}")
print("\nEDA TASK COMPLETED SUCCESSFULLY.")
