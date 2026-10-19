# CSCN8010 Lab 2: Multivariate Regression

This lab uses scikit-learn's diabetes toy dataset to model the quantitative measure of disease progression one year after baseline. It explores EDA, six BMI-only polynomial regressions (degrees 0-5), two multivariate polynomial regressions, two decision trees, two k-nearest-neighbor regressors, and two logistic classifiers for an explicitly defined high-progression screening label.

The notebook compares regression models with R-squared, MAE, and MAPE, selects models using validation data, and reserves the test set for final evaluation. Logistic regression is evaluated separately with classification metrics because it predicts a class probability rather than a continuous progression value. Results and interpretation, including limitations and error analysis, are in [lab2.ipynb](./lab2.ipynb).

With the fixed 75/10/15 split (random seed 42), the validation-selected BMI-only degree-3 model achieved test R-squared 0.207, MAE 54.63, and MAPE 48.54%. The selected multivariate degree-2 regression achieved test R-squared 0.337, MAE 45.87, and MAPE 39.47%. The supplementary logistic model achieved test F1 0.710 and ROC-AUC 0.827 for the training-median high-progression label. These results show useful but limited predictive signal and are not clinically validated.

## Setup

Use Python 3.11 or newer. From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Open `lab2.ipynb` in VS Code or Jupyter, select the `.venv` Python kernel, and run all cells from top to bottom. The checked-in CSV makes the analysis reproducible without downloading data at runtime. Recreate it from scikit-learn's bundled dataset when needed:

```powershell
python -m src.export_dataset
```

The notebook's reusable modeling helpers are in `src/analysis.py`; the original data is stored in `data/diabetes.csv`.
