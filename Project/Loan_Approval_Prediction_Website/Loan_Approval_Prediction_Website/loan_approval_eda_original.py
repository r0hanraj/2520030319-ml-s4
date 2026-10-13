# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# LOAN APPROVAL PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATASET
data = pd.read_csv("Loan_Approval_Dataset.csv")

print("=" * 50)
print("LOAN APPROVAL DATASET - EDA")
print("=" * 50)

# 2. DISPLAY FIRST 5 ROWS
print("\nFIRST 5 RECORDS")
print(data.head())

# 3. DATASET SHAPE
print("\nDATASET SHAPE")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

# 4. DATASET INFORMATION
print("\nDATASET INFORMATION")
data.info()

# 5. DATA TYPES
print("\nDATA TYPES")
print(data.dtypes)

# 6. STATISTICAL SUMMARY
print("\nSTATISTICAL SUMMARY")
print(data.describe(include="all"))

# 7. CHECK MISSING VALUES
print("\nMISSING VALUES")
print(data.isnull().sum())

# 8. MISSING VALUES VISUALIZATION
plt.figure(figsize=(10, 5))
sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Visualization")
plt.tight_layout()
plt.show()

# 9. LOAN STATUS DISTRIBUTION
plt.figure(figsize=(6, 5))
sns.countplot(data=data, x="Loan_Status")
plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.show()

# 10. GENDER VS LOAN STATUS
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Gender", hue="Loan_Status")
plt.title("Gender vs Loan Status")
plt.tight_layout()
plt.show()

# 11. MARRIED VS LOAN STATUS
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Married", hue="Loan_Status")
plt.title("Marital Status vs Loan Approval")
plt.tight_layout()
plt.show()

# 12. EDUCATION VS LOAN STATUS
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Education", hue="Loan_Status")
plt.title("Education vs Loan Approval")
plt.tight_layout()
plt.show()

# 13. SELF EMPLOYED VS LOAN STATUS
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Self_Employed", hue="Loan_Status")
plt.title("Self Employed vs Loan Approval")
plt.tight_layout()
plt.show()

# 14. PROPERTY AREA VS LOAN STATUS
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x="Property_Area", hue="Loan_Status")
plt.title("Property Area vs Loan Approval")
plt.tight_layout()
plt.show()

# 15. CREDIT HISTORY VS LOAN STATUS
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Credit_History", hue="Loan_Status")
plt.title("Credit History vs Loan Approval")
plt.tight_layout()
plt.show()

# 16. APPLICANT INCOME DISTRIBUTION
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x="ApplicantIncome", kde=True)
plt.title("Applicant Income Distribution")
plt.tight_layout()
plt.show()

# 17. CO-APPLICANT INCOME DISTRIBUTION
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x="CoapplicantIncome", kde=True)
plt.title("Co-applicant Income Distribution")
plt.tight_layout()
plt.show()

# 18. LOAN AMOUNT DISTRIBUTION
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x="LoanAmount", kde=True)
plt.title("Loan Amount Distribution")
plt.tight_layout()
plt.show()

# 19. BOX PLOTS FOR OUTLIER ANALYSIS
numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount"
]

for column in numerical_columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=data, x=column)
    plt.title(f"Box Plot of {column}")
    plt.tight_layout()
    plt.show()

# 20. CORRELATION HEATMAP
numerical_data = data.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))
sns.heatmap(
    numerical_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 21. APPLICANT INCOME VS LOAN AMOUNT
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="ApplicantIncome",
    y="LoanAmount",
    hue="Loan_Status"
)
plt.title("Applicant Income vs Loan Amount")
plt.tight_layout()
plt.show()

# 22. LOAN AMOUNT VS LOAN STATUS
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="Loan_Status", y="LoanAmount")
plt.title("Loan Amount vs Loan Status")
plt.tight_layout()
plt.show()

# 23. APPLICANT INCOME VS LOAN STATUS
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="Loan_Status", y="ApplicantIncome")
plt.title("Applicant Income vs Loan Status")
plt.tight_layout()
plt.show()

# 24. DEPENDENTS VS LOAN STATUS
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Dependents", hue="Loan_Status")
plt.title("Dependents vs Loan Approval")
plt.tight_layout()
plt.show()

# 25. FINAL EDA SUMMARY
print("\n" + "=" * 50)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 50)

print("""
Important EDA Observations:

1. The dataset contains applicant personal and financial information.
2. Loan_Status is the target variable.
3. Missing values are checked before preprocessing.
4. Credit_History can be an important feature for loan approval.
5. ApplicantIncome and LoanAmount are important financial features.
6. Categorical variables are analyzed using count plots.
7. Distribution plots help understand numerical features.
8. Box plots help identify possible outliers.
9. The correlation heatmap shows relationships between numerical variables.
""")
