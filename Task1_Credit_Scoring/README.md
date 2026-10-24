# Credit Scoring Model

A Machine Learning project developed as part of the CodeAlpha Machine Learning Internship. This project predicts the likelihood of next-month credit-card default using historical customer financial and repayment data.

## Project Objective

Build and evaluate classification models to estimate whether a customer may default on their credit-card payment in the following month.

## Dataset

- **Name:** Default of Credit Card Clients
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Records:** 30,000
- **Original columns:** 25, including the customer ID and target variable
- **Target:** `default payment next month`
- **Target values:** `0` = no default; `1` = default

The customer ID is excluded from model training because it is an identifier rather than a meaningful predictive feature.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Remove duplicate records and rows with missing target values.
3. Separate input features from the target.
4. Apply median imputation and standard scaling to numerical features.
5. Apply most-frequent imputation and one-hot encoding to selected categorical features.
6. Split the dataset into training and test sets using an 80:20 ratio with stratification.
7. Train and compare three classification algorithms.
8. Evaluate the models using accuracy, precision, recall, F1-score, and ROC-AUC.
9. Save the selected model and evaluation reports.
10. Provide an interactive prediction interface using Streamlit.

## Models Evaluated

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

The model is selected using ROC-AUC, with F1-score as the tie-breaker.

## Evaluation Results

Results from the current 80:20 stratified test split:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 67.82% | 36.77% | 63.23% | 0.4649 | 0.7106 |
| Decision Tree | 73.37% | 42.93% | 62.02% | 0.5074 | 0.7442 |
| Random Forest | 78.20% | 50.62% | 58.48% | 0.5427 | 0.7744 |

**Selected model:** Random Forest

These figures describe performance on the current test split and are not a guarantee of performance on other data.

## Project Structure

```text
Task1_Credit_Scoring/
├── app.py
├── requirements.txt
├── requirements-lock.txt
├── data/
│   └── default of credit card clients.xls
├── models/
│   └── credit_scoring_model.joblib
├── reports/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── src/
    └── train_model.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/shajeeya/codealpha_tasks.git
cd codealpha_tasks/Task1_Credit_Scoring
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Train the models

Ensure the dataset is available at `data/default of credit card clients.xls`, then run:

```powershell
python src/train_model.py
```

This creates the trained model and evaluation reports.

### 5. Launch the web application

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal.

## Generated Outputs

- `models/credit_scoring_model.joblib` — selected trained pipeline
- `reports/metrics.json` — evaluation metrics and model details
- `reports/model_comparison.csv` — model comparison table
- `reports/confusion_matrix.png` — confusion matrix for the selected model
- `reports/roc_curve.png` — ROC curve for the selected model

## Limitations and Responsible Use

- This project uses historical credit-card client data from Taiwan and may not generalize to other populations or lending environments.
- The default class has lower precision and recall than the non-default class in the current evaluation.
- Model predictions and probabilities are estimates, not certainties.
- The interface is an educational demonstration, not a validated credit assessment tool. It should not be used to approve or reject real credit applications.
- The dataset contains demographic features. Their use raises fairness and regulatory considerations that would require additional analysis before any real-world application.

## Internship Task

**Program:** CodeAlpha Machine Learning Internship  
**Task:** Credit Scoring Model

This project demonstrates supervised classification, data preprocessing, model comparison, evaluation, model persistence, and deployment through a Streamlit application.

## License and Dataset Attribution

The dataset is provided by the UCI Machine Learning Repository under the license indicated on its official dataset page. Refer to the original source for attribution and usage terms.
