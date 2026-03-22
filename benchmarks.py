import numpy as np
import pandas as pd
from typing import Callable

import load_data   
import poos
import autoregressive

API_KEY = "cd30e8d67ebd36672d4b0ebfc5069427"

# ── 1. Load data ──────────────────────────────────────────────────────────────
y_series = load_data.load_transformed_series_latest_release("INDPRO", API_KEY)

# ── 2. Prepare data ───────────────────────────────────────────────────────────
X_df = pd.DataFrame({
    "lag_1": y_series.shift(1),
    "lag_2": y_series.shift(2)
})

# Align and drop NaNs together
df = pd.concat([X_df, y_series], axis=1).dropna()
X_ar = df.iloc[:, :-1].reset_index(drop=True)
y_ar = df.iloc[:,  -1].reset_index(drop=True)

# ── 3. Run POOS with AR model ─────────────────────────────────────────────────
X_out, y_out, rmse, mae = poos.poos_validation(
    method=autoregressive.ar_model_nowcast,
    X=X_ar,
    y=y_ar,
    prop_train=0.9,        # ← matches poos.py parameter name
)

# ── 4. Results ────────────────────────────────────────────────────────────────
print("=== POOS Results (first 10 rows) ===")
print(y_out.head(10))
print(f"\nRMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"OOS observations: {len(y_out)}")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
poos.plot_poos_results(y_ar, y_out, title="INDPRO — AR Model POOS")