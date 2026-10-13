from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)

DATA_PATH = "data/Loan_Approval_Dataset.csv"

# -----------------------------
# Load data and train the same
# hard-voting ensemble used in
# the supplied project code.
# -----------------------------
data = pd.read_csv(DATA_PATH)
data = data.drop(columns=["Loan_ID"], errors="ignore")

X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"].map({"Y": 1, "N": 0})

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

def make_pipeline(model):
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

base_models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(random_state=42)
    ),
    "Random Forest": make_pipeline(
        RandomForestClassifier(n_estimators=200, random_state=42)
    ),
    "Gradient Boosting": make_pipeline(
        GradientBoostingClassifier(n_estimators=100, random_state=42)
    ),
}

model_metrics = {}
for name, model in base_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    model_metrics[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }

ensemble_model = VotingClassifier(
    estimators=[
        ("lr", base_models["Logistic Regression"]),
        ("dt", base_models["Decision Tree"]),
        ("rf", base_models["Random Forest"]),
        ("gb", base_models["Gradient Boosting"]),
    ],
    voting="hard"
)
ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)

ensemble_metrics = {
    "accuracy": accuracy_score(y_test, ensemble_pred),
    "precision": precision_score(y_test, ensemble_pred, zero_division=0),
    "recall": recall_score(y_test, ensemble_pred, zero_division=0),
    "f1": f1_score(y_test, ensemble_pred, zero_division=0),
}
cm = confusion_matrix(y_test, ensemble_pred).tolist()

# Dataset information used by the dashboard
dataset_info = {
    "rows": int(data.shape[0]),
    "features": int(X.shape[1]),
    "approved": int((y == 1).sum()),
    "rejected": int((y == 0).sum()),
    "missing": int(data.isnull().sum().sum()),
    "approval_rate": float((y == 1).mean() * 100),
}

def clean_number(value):
    if value is None or value == "":
        return np.nan
    return float(value)

def build_applicant(payload):
    return pd.DataFrame([{
        "Gender": payload.get("Gender"),
        "Married": payload.get("Married"),
        "Dependents": payload.get("Dependents"),
        "Education": payload.get("Education"),
        "Self_Employed": payload.get("Self_Employed"),
        "ApplicantIncome": clean_number(payload.get("ApplicantIncome")),
        "CoapplicantIncome": clean_number(payload.get("CoapplicantIncome")),
        "LoanAmount": clean_number(payload.get("LoanAmount")),
        "Loan_Amount_Term": clean_number(payload.get("Loan_Amount_Term")),
        "Credit_History": clean_number(payload.get("Credit_History")),
        "Property_Area": payload.get("Property_Area"),
    }])

@app.route("/")
def home():
    model_rows = [
        {"name": name, **{k: round(v * 100, 2) for k, v in metrics.items()}}
        for name, metrics in model_metrics.items()
    ]
    model_rows.append({
        "name": "Voting Ensemble",
        **{k: round(v * 100, 2) for k, v in ensemble_metrics.items()}
    })
    return render_template(
        "index.html",
        dataset=dataset_info,
        ensemble=ensemble_metrics,
        models=model_rows,
        confusion=cm,
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        applicant = build_applicant(payload)

        prediction = int(ensemble_model.predict(applicant)[0])

        # Hard voting: show how many base models voted for the predicted class.
        votes = {}
        vote_values = []
        for name, model in base_models.items():
            vote = int(model.predict(applicant)[0])
            votes[name] = "Approved" if vote == 1 else "Rejected"
            vote_values.append(vote)

        agreement = sum(v == prediction for v in vote_values) / len(vote_values) * 100
        status = "APPROVED" if prediction == 1 else "REJECTED"

        return jsonify({
            "success": True,
            "status": status,
            "prediction": prediction,
            "agreement": round(agreement, 1),
            "votes": votes,
            "note": "This is a machine-learning prediction for academic demonstration, not a lending decision."
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

@app.route("/api/summary")
def summary():
    return jsonify({
        "dataset": dataset_info,
        "ensemble": {k: round(v * 100, 2) for k, v in ensemble_metrics.items()},
        "models": [
            {"name": name, **{k: round(v * 100, 2) for k, v in metrics.items()}}
            for name, metrics in model_metrics.items()
        ] + [{
            "name": "Voting Ensemble",
            **{k: round(v * 100, 2) for k, v in ensemble_metrics.items()}
        }],
        "confusion_matrix": cm
    })

if __name__ == "__main__":
    print("\nLoan Approval Prediction Website")
    print(f"Dataset: {dataset_info['rows']} rows x {dataset_info['features']} features")
    print(f"Voting Ensemble accuracy: {ensemble_metrics['accuracy'] * 100:.2f}%")
    print("Open http://127.0.0.1:5000 in your browser.\n")
    app.run(debug=True)
