# Heart Disease Prediction

A machine learning project developed as part of the CodeAlpha Machine Learning Internship. This project compares classification models to predict the presence or absence of heart disease using the Cleveland Heart Disease dataset.

## Overview

The project uses clinical attributes to train and evaluate three binary classification models:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest

The original dataset contains 303 records and 13 input features. Missing values in the `ca` and `thal` columns are handled using median imputation within the model pipeline.

The original target values are converted to binary labels:

- `0` — No disease
- `1–4` — Disease present

## Technologies

- Python
- NumPy
- Pandas
- scikit-learn
- SciPy
- Matplotlib
- Joblib
- Pytest

## Repository Structure

Task4_Heart_Disease_Prediction/
├── data/
│   └── processed.cleveland.data
├── models/
│   └── heart_disease_model.joblib
├── results/
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   ├── model_metadata.json
│   └── test_predictions.csv
├── src/
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── predict.py
│   └── test_model.py
├── requirements.txt
└── README.md

## Dataset

The project uses the Cleveland subset of the UCI Heart Disease dataset.

Dataset source: https://archive.ics.uci.edu/dataset/45/heart+disease

The dataset has 303 records, 13 input features, and one target column. Six records contain missing-value markers in the `ca` and `thal` fields.

The dataset is intended for educational and research use. Refer to the original source for its terms of use.

## Setup

Python 3.12 is recommended for the documented environment.

1. Clone the repository and navigate to this project directory.
2. Create and activate a virtual environment.
3. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train and Evaluate

Run the training script from the repository root:

```bash
python Task4_Heart_Disease_Prediction/src/train_model.py
```

The script compares the candidate models using five-fold stratified cross-validation on the training split, selects the model with the highest mean cross-validation F1-score, evaluates it on a held-out test set, and saves the fitted pipeline and evaluation artifacts.

## Make a Prediction

The prediction script accepts 13 numeric values in this order:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

Example command:

```bash
python Task4_Heart_Disease_Prediction/src/predict.py 63 1 1 145 233 1 2 150 0 2.3 3 0 6
```

The saved model must exist in the `models` directory before predictions can be made. The script reports the binary prediction and the model's estimated probability for the positive class.

## Evaluation

The initial Colab experiment selected Logistic Regression based on mean cross-validation F1-score.

Initial held-out test results for that experiment:

| Metric | Result |
|---|---:|
| Accuracy | 86.89% |
| Balanced accuracy | 87.34% |
| Precision — disease present | 81.25% |
| Recall — disease present | 92.86% |
| F1-score — disease present | 86.67% |
| ROC-AUC | 95.13% |

These figures describe the initial Colab experiment only. The local training script must be executed successfully before its generated results can be treated as verified and reproducible.

See the `results/` directory for the model comparison, predictions, metadata, and confusion matrix.

## Limitations and Disclaimer

- This dataset is small and historical; its results may not generalize to current clinical populations.
- Model predictions are not medical diagnoses and must not be used to make treatment decisions.
- The reported probability is not a clinically calibrated risk estimate.
- Evaluation on a single held-out split can vary with the data split and model configuration.

## Author

Developed as part of the CodeAlpha Machine Learning Internship.