# =============================================================
# model_3_final.py — best model selection + metrics + correlation bar plot
# =============================================================

import os
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ------------------------------------------------
# Utility functions
# ------------------------------------------------
def _rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def _plot_correlation_bar(X, y, target_name):
    corrs = X.join(y).corr()[target_name].drop(target_name)
    plt.figure(figsize=(7,4))
    corrs.plot(kind='bar')
    plt.title(f"Correlation of {target_name} with other features")
    plt.ylabel("Correlation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# Main model runner
# ------------------------------------------------
def run_model_3_final(preprocessed_dict, show_plots=True):

    TARGETS_LIST = [
        'Ammonia (mg/l)',
        'Biochemical Oxygen Demand (mg/l)',
        'Dissolved Oxygen (mg/l)',
        'Orthophosphate (mg/l)',
        'pH (ph units)',
        'Temperature (cel)',
        'Nitrogen (mg/l)',
        'Nitrate (mg/l)'
    ]

    # -------------------------------
    # Allowed models ONLY
    # -------------------------------
    constructors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=60, n_jobs=-1, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=60, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=60, learning_rate=0.1,
            max_depth=3, subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42
        ),
    }

    models_to_use = list(constructors.keys())  # only these models

    per_target_summary = []

    # ===============================================================
    # Run Prediction for Each Target Variable
    # ===============================================================
    for target, data in tqdm(preprocessed_dict.items(), desc="Targets"):
        print("\n" + "="*60)
        print(f"🔹 Predicting: {target}")
        print("="*60)

        X_train, X_val, X_test, y_train, y_val, y_test = data

        # Only keep other 7 targets as features
        cols_to_use = [c for c in TARGETS_LIST if c != target]
        X_train = X_train[cols_to_use]
        X_val   = X_val[cols_to_use]
        X_test  = X_test[cols_to_use]

        # ------------------------------------------------------
        #  Correlation plot
        # ------------------------------------------------------
        if show_plots:
            _plot_correlation_bar(X_train, y_train, target)

        # ------------------------------------------------------
        #  Train allowed models & select best
        # ------------------------------------------------------
        models_summary = {}
        detailed_models = {}

        for name, model in constructors.items():

            if name not in models_to_use:
                continue

            try:
                model.fit(X_train, y_train)
                preds_val = model.predict(X_val)
                val_rmse = _rmse(y_val, preds_val)

                models_summary[name] = val_rmse
                detailed_models[name] = model

            except Exception as e:
                print(f"❌ {name} failed: {e}")
                models_summary[name] = np.nan
                detailed_models[name] = None

        # Safety: ensure at least one valid model exists
        valid = {k: v for k, v in models_summary.items() if not np.isnan(v)}
        if len(valid) == 0:
            print(f"❌ No valid model for target {target}. Skipping.")
            continue

        # Select best model = lowest validation RMSE
        best_name = min(valid, key=valid.get)
        best_model = detailed_models[best_name]

        print(f"\n✅ Best model for {target}: {best_name}")

        # ------------------------------------------------------
        #  Compute & print metrics
        # ------------------------------------------------------
        def compute_metrics(model, X, y):
            y_pred = model.predict(X)
            return (
                _rmse(y, y_pred),
                mean_absolute_error(y, y_pred),
                r2_score(y, y_pred),
            )

        train_rmse, train_mae, train_r2 = compute_metrics(best_model, X_train, y_train)
        val_rmse, val_mae, val_r2       = compute_metrics(best_model, X_val, y_val)
        test_rmse, test_mae, test_r2    = compute_metrics(best_model, X_test, y_test)

        print(f"📊 TRAIN — RMSE={train_rmse:.3f}, MAE={train_mae:.3f}, R²={train_r2:.3f}")
        print(f"📊 VAL   — RMSE={val_rmse:.3f}, MAE={val_mae:.3f}, R²={val_r2:.3f}")
        print(f"📊 TEST  — RMSE={test_rmse:.3f}, MAE={test_mae:.3f}, R²={test_r2:.3f}")

        # Save summary
        per_target_summary.append({
            "Target": target,
            "Best Model": best_name,
            "Train RMSE": train_rmse,
            "Train MAE": train_mae,
            "Train R2": train_r2,
            "Val RMSE": val_rmse,
            "Val MAE": val_mae,
            "Val R2": val_r2,
            "Test RMSE": test_rmse,
            "Test MAE": test_mae,
            "Test R2": test_r2,
        })

    # ------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------
    final_df = pd.DataFrame(per_target_summary)
    print("\n=== FINAL RESULTS ===")
    display(final_df)

    return final_df
