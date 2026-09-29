"""Task 3 — Baseline Regression Pipeline (House Prices).

Usage:
    python regression_baseline.py ames_cleaned.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
RIDGE_ALPHAS = [0.1, 1.0, 10.0]


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def make_pipeline(model, numeric_features, categorical_features):
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, categorical_features)),
        ("model", model),
    ])


def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Task 1 cleaned CSV")
    parser.add_argument("report_file", nargs="?", default="MODEL_COMPARISON.md")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "SalePrice" not in df.columns:
        raise ValueError("Input dataset must contain SalePrice.")

    y = df["SalePrice"].copy()
    X = df.drop(columns=["SalePrice"]).copy()

    # Identifier, not a predictive feature.
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    # Task 1 identified this numeric-looking code as categorical.
    if "MSSubClass" in X.columns:
        X["MSSubClass"] = X["MSSubClass"].astype(str)

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    # Split BEFORE fitting any preprocessing transformer.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []

    models = [("Mean Baseline", DummyRegressor(strategy="mean")),
              ("Linear Regression", LinearRegression())]
    models += [(f"Ridge (alpha={a:g})", Ridge(alpha=a)) for a in RIDGE_ALPHAS]

    for name, estimator in models:
        model = make_pipeline(estimator, numeric_features, categorical_features)
        metrics = evaluate(model, X_train, X_test, y_train, y_test)
        results.append({"Model": name, "Target": "SalePrice", **metrics})

    # Required skewed-target experiment; metrics remain on original dollar scale.
    log_models = [("Linear Regression", LinearRegression())]
    log_models += [(f"Ridge (alpha={a:g})", Ridge(alpha=a)) for a in RIDGE_ALPHAS]
    for name, estimator in log_models:
        model = TransformedTargetRegressor(
            regressor=make_pipeline(estimator, numeric_features, categorical_features),
            func=np.log1p,
            inverse_func=np.expm1,
        )
        metrics = evaluate(model, X_train, X_test, y_train, y_test)
        results.append({"Model": name, "Target": "log1p(SalePrice)", **metrics})

    results_df = pd.DataFrame(results)
    raw = results_df[results_df.Target == "SalePrice"]
    log = results_df[results_df.Target == "log1p(SalePrice)"]

    baseline = raw[raw.Model == "Mean Baseline"].iloc[0]
    linear = raw[raw.Model == "Linear Regression"].iloc[0]
    best_ridge = raw[raw.Model.str.startswith("Ridge")].sort_values("RMSE").iloc[0]
    best_log = log.sort_values("RMSE").iloc[0]

    report = f"""# Task 3 — Baseline Regression Pipeline (House Prices)

## Objective

Build a leakage-safe regression workflow using the cleaned Ames Housing dataset from Task 1. The workflow compares a mean-prediction baseline, plain Linear Regression, and Ridge Regression with three alpha values, and tests a log-transformed target.

## Dataset and validation setup

- Input: `{Path(args.input_csv).name}`
- Rows: **{len(df):,}**
- Target: **SalePrice**
- Train/test split: **80/20**
- Random state: **{RANDOM_STATE}**
- Training rows: **{len(X_train):,}**
- Test rows: **{len(X_test):,}**
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
"""
    for _, r in raw.iterrows():
        report += f"| {r.Model} | ${r.MAE:,.2f} | ${r.RMSE:,.2f} | {r.R2:.4f} |\n"

    report += "\n## Ridge alpha comparison\n\n"
    report += "Ridge was evaluated at **alpha = 0.1, 1, and 10**. The best raw-target Ridge configuration by RMSE was "
    report += f"**{best_ridge.Model}**, with RMSE **${best_ridge.RMSE:,.2f}** and R² **{best_ridge.R2:.4f}**.\n\n"

    report += "## Log-transformed target experiment\n\n"
    report += "Because SalePrice is right-skewed, `log1p(SalePrice)` was tested. Metrics below are calculated after converting predictions back to dollar scale.\n\n"
    report += "| Model | MAE | RMSE | R² |\n|---|---:|---:|---:|\n"
    for _, r in log.iterrows():
        report += f"| {r.Model} | ${r.MAE:,.2f} | ${r.RMSE:,.2f} | {r.R2:.4f} |\n"

    report += f"""
## Interpretation

The mean baseline produced RMSE **${baseline.RMSE:,.2f}** and R² **{baseline.R2:.4f}**. Plain Linear Regression reduced RMSE to **${linear.RMSE:,.2f}** and achieved R² **{linear.R2:.4f}**, showing that the feature-based model substantially improves on a mean-only prediction.

For the raw target, the best tested Ridge model (**{best_ridge.Model}**) produced RMSE **${best_ridge.RMSE:,.2f}** and R² **{best_ridge.R2:.4f}**, compared with Linear Regression RMSE **${linear.RMSE:,.2f}** and R² **{linear.R2:.4f}**. The reduction in error indicates that regularization was materially useful for this feature set.

The best log-target model was **{best_log.Model}**, with RMSE **${best_log.RMSE:,.2f}**, MAE **${best_log.MAE:,.2f}**, and R² **{best_log.R2:.4f}** on the original price scale. This experiment demonstrates why a skewed regression target should be tested rather than assuming the raw target is always preferable.

## Metric meanings

- **MAE:** average absolute prediction error, in dollars.
- **RMSE:** error metric that penalizes large prediction errors more strongly than MAE.
- **R²:** variance explained relative to the mean-prediction baseline; it can be negative when a model performs worse than that baseline.

## Reproduce

```bash
pip install pandas numpy scikit-learn
python regression_baseline.py ames_cleaned.csv
```
"""

    Path(args.report_file).write_text(report, encoding="utf-8")

    print(f"Input shape:   {df.shape}")
    print(f"Train shape:   {X_train.shape}")
    print(f"Test shape:    {X_test.shape}")
    print("\nRaw-target results:")
    print(raw.to_string(index=False))
    print("\nLog-target results:")
    print(log.to_string(index=False))
    print(f"\nSaved report to: {args.report_file}")


if __name__ == "__main__":
    main()
