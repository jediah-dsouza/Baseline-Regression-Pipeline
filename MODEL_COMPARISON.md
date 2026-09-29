# Task 3 — Baseline Regression Pipeline (House Prices)

## Objective

Build a leakage-safe regression workflow using the cleaned Ames Housing dataset from Task 1. The workflow compares a mean-prediction baseline, plain Linear Regression, and Ridge Regression with three alpha values, and tests a log-transformed target.

## Dataset and validation setup

- Input: `ames_cleaned.csv`
- Rows: **1,460**
- Target: **SalePrice**
- Train/test split: **80/20**
- Random state: **42**
- Training rows: **1,168**
- Test rows: **292**
- `Id` was removed because it is an identifier rather than a predictive measurement.
- `MSSubClass` was explicitly converted to categorical, following the Task 1 type decision.

## Leakage prevention

The train/test split occurs **before** preprocessing is fitted. Imputation, one-hot encoding, and scaling are all inside a scikit-learn `Pipeline`/`ColumnTransformer`, so they are fitted only on the training partition and then applied unchanged to the test partition.

- Numeric: median imputation → StandardScaler
- Categorical: most-frequent imputation → OneHotEncoder(handle_unknown="ignore")
- No target information is used by preprocessing.
- The log-target experiment uses `log1p(SalePrice)` inside `TransformedTargetRegressor`; predictions are converted back with `expm1` before metrics are calculated.

## Main comparison — raw SalePrice

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | $62,575.93 | $87,619.03 | -0.0009 |
| Linear Regression | $21,018.82 | $66,962.22 | 0.4154 |
| Ridge (alpha=0.1) | $18,076.07 | $28,077.74 | 0.8972 |
| Ridge (alpha=1) | $18,971.94 | $28,901.68 | 0.8911 |
| Ridge (alpha=10) | $19,193.97 | $29,763.67 | 0.8845 |

## Ridge alpha comparison

Ridge was evaluated at **alpha = 0.1, 1, and 10**. The best raw-target Ridge configuration by RMSE was **Ridge (alpha=0.1)**, with RMSE **$28,077.74** and R² **0.8972**.

## Log-transformed target experiment

Because SalePrice is right-skewed, `log1p(SalePrice)` was tested. Metrics below are calculated after converting predictions back to dollar scale.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | $15,233.69 | $23,730.52 | 0.9266 |
| Ridge (alpha=0.1) | $15,313.25 | $23,637.57 | 0.9272 |
| Ridge (alpha=1) | $16,213.25 | $25,218.70 | 0.9171 |
| Ridge (alpha=10) | $16,725.97 | $27,028.95 | 0.9048 |

## Interpretation

The mean baseline produced RMSE **$87,619.03** and R² **-0.0009**. Plain Linear Regression reduced RMSE to **$66,962.22** and achieved R² **0.4154**, showing that the feature-based model substantially improves on a mean-only prediction.

For the raw target, the best tested Ridge model (**Ridge (alpha=0.1)**) produced RMSE **$28,077.74** and R² **0.8972**, compared with Linear Regression RMSE **$66,962.22** and R² **0.4154**. The reduction in error indicates that regularization was materially useful for this feature set.

The best log-target model was **Ridge (alpha=0.1)**, with RMSE **$23,637.57**, MAE **$15,313.25**, and R² **0.9272** on the original price scale. This experiment demonstrates why a skewed regression target should be tested rather than assuming the raw target is always preferable.

## Metric meanings

- **MAE:** average absolute prediction error, in dollars.
- **RMSE:** error metric that penalizes large prediction errors more strongly than MAE.
- **R²:** variance explained relative to the mean-prediction baseline; it can be negative when a model performs worse than that baseline.

## Reproduce

```bash
pip install pandas numpy scikit-learn
python regression_baseline.py ames_cleaned.csv
```
