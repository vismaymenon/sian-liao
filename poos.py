# poos.py

import numpy as np
import pandas as pd
from typing import Callable
import load_data

def print_results(dict):
    for key, value in dict.items():
        print(f"{key}: {value}")

# ── Benchmark model ───────────────────────────────────────────────────────────

def placeholder_model(X, y):
    """
    Benchmark (unconditional mean) model.
    Treats the last row of X and last element of y as the test observation.

    Inputs:
        X : pd.DataFrame, shape (t+1, n_features)  — last row is test
        y : pd.Series,    shape (t+1,)              — last element is test

    Outputs:
        X_train          (pd.DataFrame) : training X
        y_actual         (np.ndarray)   : training y
        y_train_predicted(np.ndarray)   : in-sample predictions (all = mean)
        X_test           (pd.DataFrame) : test X (last row of input X)
        y_test_actual    (float)        : held-out true value
        y_test_predicted (float)        : prediction = mean of training y
    """
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1].values
    X_test = X.iloc[[-1]]
    y_test_actual = float(y.iloc[-1])

    y_train_mean = float(np.mean(y_train))
    y_train_predicted = y_train_mean
    y_test_predicted  = y_train_mean

    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_train_predicted": y_train_predicted,
        "X_test": X_test,
        "y_test_actual": y_test_actual,
        "y_test_predicted": y_test_predicted
    }

# ── POOS ──────────────────────────────────────────────────────────────────────

def poos_validation(
    method: Callable,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    prop_train: float = 0.9,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Pseudo Out-of-Sample (POOS) expanding-window validation.

    Parameters
    ----------
    method         : Callable  — signature: method(X, y) -> (coefficients, X, y_actual,
                                  y_train_predicted, y_test_actual, y_test_predicted)
    X              : feature matrix (n_samples, n_features)
    y              : target array   (n_samples,)
    min_train_size : if float in (0,1), treated as proportion of sample;
                     if int >= 2, treated as absolute number of training obs.
                     Defaults to 0.9 (90% burn-in).
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    n = len(y)

    train_size = int(prop_train * n) if isinstance(prop_train, float) else 100 # with min train size as 100
    test_indices, actuals = [], []
    preds_point, preds_50_lower, preds_50_upper, preds_80_lower, preds_80_upper = [], [], [], [], []

    for t in range(n - train_size):
        X_window = X.iloc[t:t+train_size+1]
        y_window = y.iloc[t:t+train_size+1]

        _, y_train_actual, y_train_predicted, _, y_test_actual, y_test_predicted = method(X_window, y_window).values()
        std_error = np.std(y_train_actual - y_train_predicted)

        test_indices.append(t)
        actuals.append(float(y_test_actual))
        preds_point.append(float(y_test_predicted))
        preds_50_lower.append(float(y_test_predicted) - 0.674 * std_error)
        preds_50_upper.append(float(y_test_predicted) + 0.674 * std_error)
        preds_80_lower.append(float(y_test_predicted) - 1.282 * std_error)
        preds_80_upper.append(float(y_test_predicted) + 1.282 * std_error)

    y_df = pd.DataFrame(
        index=test_indices,
        data={
            "y_true": actuals,
            "y_hat": preds_point,
            "pred_50_lower": preds_50_lower,
            "pred_50_upper": preds_50_upper,
            "pred_80_lower": preds_80_lower,
            "pred_80_upper": preds_80_upper,
        }
    )

    rmse = np.sqrt(np.mean((y_df["y_true"] - y_df["y_hat"]) ** 2))
    mae  = np.mean(np.abs(y_df["y_true"] - y_df["y_hat"]))

    return X.iloc[test_indices].copy(), y_df, rmse, mae


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    API_KEY = "cd30e8d67ebd36672d4b0ebfc5069427"

    # Load & transform a FRED series as target (y)
    # Using INDPRO (Industrial Production) as an example
    y_series = load_data.load_transformed_series_latest_release("INDPRO", API_KEY)

    # Use lags of y as a simple feature matrix (AR-style)
    X_df = pd.DataFrame({
        "lag_1": y_series.shift(1),
        "lag_2": y_series.shift(2),
        "lag_3": y_series.shift(3),
    })

    # Align and drop NaNs
    df = pd.concat([X_df, y_series], axis=1).dropna()
    X = df.iloc[:, :-1].reset_index(drop=True)
    y = df.iloc[:, -1].reset_index(drop=True)

    print(f"Sample size: {len(y)}")
    print(f"Features:    {X.columns.tolist()}\n")

    # Run POOS with benchmark model
    X_out, y_out, rmse, mae = poos_validation(
        method=placeholder_model,
        X=X,
        y=y,
        prop_train=0.9,
    )

    print("=== POOS Results (first 5 rows) ===")
    print(y_out.head())

    rmse = np.sqrt(np.mean((y_out["y_true"] - y_out["y_hat"]) ** 2))
    mae  = np.mean(np.abs(y_out["y_true"] - y_out["y_hat"]))
    print(f"\nOut-of-sample RMSE : {rmse:.6f}")
    print(f"Out-of-sample MAE  : {mae:.6f}")
    print(f"\nOOS observations   : {len(y_out)}")