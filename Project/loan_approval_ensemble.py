# ============================================================
# LOAN APPROVAL PREDICTION USING ENSEMBLE MACHINE LEARNING
# Dataset: Loan_Approval_Dataset.csv
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. LOAD DATASET
data = pd.read_csv("Loan_Approval_Dataset.csv")

print("=" * 45)
print("LOAN APPROVAL PREDICTION")
print("=" * 45)

print("\nDataset Loaded Successfully!")
print("\nFirst 5 Records:")
print(data.head())

print("\nDataset Shape:", data.shape)

print("\nMissing Values:")
print(data.isnull().sum())


# 2. REMOVE LOAN ID
if "Loan_ID" in data.columns:
    data = data.drop("Loan_ID", axis=1)


# 3. DEFINE INPUT AND OUTPUT
X = data.drop("Loan_Status", axis=1)

y = data["Loan_Status"].map({
    "Y": 1,
    "N": 0
})


# 4. IDENTIFY FEATURE TYPES
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


# 5. DATA PREPROCESSING
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# 6. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)


# 7. CREATE MODELS
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

dt_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

gb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(n_estimators=100, random_state=42))
])


# 8. TRAIN AND COMPARE INDIVIDUAL MODELS
models = {
    "Logistic Regression": lr_pipeline,
    "Decision Tree": dt_pipeline,
    "Random Forest": rf_pipeline,
    "Gradient Boosting": gb_pipeline
}

results = {}

print("\n" + "=" * 45)
print("TRAINING INDIVIDUAL MODELS")
print("=" * 45)

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print(f"{name}: {accuracy * 100:.2f}%")


# 9. CREATE VOTING ENSEMBLE
ensemble_model = VotingClassifier(
    estimators=[
        ("lr", lr_pipeline),
        ("dt", dt_pipeline),
        ("rf", rf_pipeline),
        ("gb", gb_pipeline)
    ],
    voting="hard"
)

print("\n" + "=" * 45)
print("TRAINING VOTING ENSEMBLE MODEL")
print("=" * 45)

ensemble_model.fit(X_train, y_train)

ensemble_predictions = ensemble_model.predict(X_test)

ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

results["Voting Ensemble"] = ensemble_accuracy

print(f"Voting Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%")


# 10. MODEL COMPARISON
print("\n" + "=" * 45)
print("MODEL ACCURACY COMPARISON")
print("=" * 45)

for name, accuracy in results.items():
    print(f"{name}: {accuracy * 100:.2f}%")


best_model = max(results, key=results.get)

print("\nBest Model:", best_model)
print(f"Best Accuracy: {results[best_model] * 100:.2f}%")


# 11. CLASSIFICATION REPORT
print("\n" + "=" * 45)
print("CLASSIFICATION REPORT - VOTING ENSEMBLE")
print("=" * 45)

print(classification_report(
    y_test,
    ensemble_predictions,
    target_names=["Rejected", "Approved"]
))


# 12. CONFUSION MATRIX
cm = confusion_matrix(y_test, ensemble_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"]
)

plt.title("Confusion Matrix - Loan Approval Prediction")
plt.xlabel("Predicted Loan Status")
plt.ylabel("Actual Loan Status")
plt.tight_layout()
plt.show()


# 13. ACCURACY COMPARISON GRAPH
plt.figure(figsize=(10, 5))

plt.bar(
    list(results.keys()),
    [accuracy * 100 for accuracy in results.values()]
)

plt.title("Accuracy Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# 14. PREDICT A NEW LOAN APPLICATION
new_applicant = pd.DataFrame([{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}])

prediction = ensemble_model.predict(new_applicant)

print("\n" + "=" * 45)
print("NEW LOAN APPLICATION PREDICTION")
print("=" * 45)

if prediction[0] == 1:
    print("Loan Status: APPROVED")
else:
    print("Loan Status: REJECTED")
