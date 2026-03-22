import numpy as np
import pandas as pd
from typing import Callable
import load_data   
import poos
import autoregressive

# poos.py  (add this to the bottom / run section)

import load_data
import autoregressive

API_KEY = "your_api_key_here"

# ── 1. Load data ──────────────────────────────────────────────────────────────
y_series = load_data.load_transformed_series_latest_release("INDPRO", API_KEY)

# Simple AR feature matrix (lags of y)
X_df = pd.DataFrame({
    "lag_1": y_series.shift(1),
    "lag_2": y_series.shift(2),
})

# Align and drop NaNs
df = pd.concat([X_df, y_series], axis=1).dropna()
X = df.iloc[:, :-1].reset_index(drop=True)
y = df.iloc[:,  -1].reset_index(drop=True)

# ── 2. Wrapper to match poos_validation's expected signature ──────────────────
def ar_wrapper(X_window, y_window):
    """
    Adapts ar_model_nowcast (which only needs y) to the poos_validation signature.
    poos passes (X_window, y_window) where last row = test observation.
    """
    result = autoregressive.ar_model_nowcast(y_window)

    return (
        result["coefficients"],
        result["X"],
        result["y_actual"],
        result["y_train_predicted"],
        result["y_test_actual"],
        result["y_test_predicted"],
    )

# ── 3. Run POOS ───────────────────────────────────────────────────────────────
X_out, y_pred, y_actual_df = poos_validation(
    method=ar_wrapper,
    X=X,
    y=y,
    min_train_size=0.9,
)

# ── 4. Results ────────────────────────────────────────────────────────────────
print(y_actual_df.head(10))

rmse = np.sqrt(np.mean((y_actual_df["y_true"] - y_actual_df["y_hat"]) ** 2))
mae  = np.mean(np.abs(y_actual_df["y_true"] - y_actual_df["y_hat"]))
print(f"\nRMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"OOS observations: {len(y_actual_df)}")