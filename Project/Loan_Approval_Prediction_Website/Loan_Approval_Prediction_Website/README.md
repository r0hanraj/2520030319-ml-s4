# Loan Approval Prediction Using Ensemble Machine Learning — Web App

A VS Code-ready Flask website built from the supplied project materials and dataset.

## What this website does

- Loads the supplied `Loan_Approval_Dataset.csv` (1,000 records, 12 input features + target).
- Uses the same preprocessing and hard-voting ensemble approach as the supplied `loan_approval_ensemble.py`.
- Trains:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Voting Ensemble
- Shows dataset statistics and test-set model metrics.
- Provides an interactive loan application form.
- Returns an Approved/Rejected prediction and the percentage of base-model votes agreeing with the ensemble.
- Includes a model-comparison section and confusion matrix.
- Keeps the original supplied EDA/model scripts in the project folder for reference.

## Important project-source note

The presentation/report describes a more advanced Random Forest + XGBoost + LightGBM + stacking architecture, while the supplied executable Python script implements Logistic Regression + Decision Tree + Random Forest + Gradient Boosting with a hard VotingClassifier. This website follows the supplied executable Python implementation so the web prediction is reproducible from the code you uploaded.

## Requirements

- Python 3.10 or newer
- VS Code
- Internet is only needed once to install Python packages.

## Run in VS Code — Windows

1. Extract this ZIP.
2. Open the extracted `Loan_Approval_Prediction_Website` folder in VS Code.
3. Open **Terminal > New Terminal**.
4. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

6. Install packages:

   ```powershell
   pip install -r requirements.txt
   ```

7. Start the website:

   ```powershell
   python app.py
   ```

8. Open:

   `http://127.0.0.1:5000`

## Run in VS Code — macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Stop the website

In the terminal press `Ctrl+C`.

## Project structure

```text
Loan_Approval_Prediction_Website/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── loan_approval_ensemble_original.py
├── loan_approval_eda_original.py
├── data/
│   └── Loan_Approval_Dataset.csv
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── app.js
```

## Academic/demo disclaimer

The output is a machine-learning prediction based on the supplied historical dataset. It is intended for the academic project demonstration and should not be treated as a real-world lending decision or financial advice.

## Troubleshooting

### `python` is not recognized

Install Python and enable **Add Python to PATH** during installation. Restart VS Code afterward.

### `ModuleNotFoundError: flask`

Make sure the virtual environment is activated and run:

```bash
pip install -r requirements.txt
```

### Port 5000 is already in use

Change the last line in `app.py` to:

```python
app.run(debug=True, port=5001)
```

Then open `http://127.0.0.1:5001`.

### Model takes a little time on first start

That is expected: the website trains the ensemble when Flask starts.
