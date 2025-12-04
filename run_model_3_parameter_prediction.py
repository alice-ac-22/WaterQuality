# run_model_3_parameter_prediction_selected.py
import time
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------- Helper functions ----------
def _plot_pred_vs_residuals(y_true, y_pred, target_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], color="red", linestyle="--")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"{target_name} — Predicted vs Actual")
    axes[0].grid(True)

    # Residuals
    axes[1].hist(residuals, bins=25, edgecolor="k", alpha=0.6)
    axes[1].set_title(f"{target_name} — Residuals (mean={residuals.mean():.3g})")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# ---------- Main function ----------
def run_model_3_parameter_prediction(preprocessed_dict):
    """
    Run Model 3 (Parameter-to-Parameter Prediction) with only selected models
    and display all plots (no saving to files).

    Returns:
        full_summary_df (pd.DataFrame)
        best_models_per_target (dict)
    """
    per_target_summaries = []
    best_models_per_target = {}

    # ------------------------------
    # Only selected models
    # ------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=60, n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=60, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=60, verbosity=0, random_state=42),
    }

    for target, data_tuple in tqdm(preprocessed_dict.items(), desc="Parameters"):
        print("\n" + "="*60)
        print(f"🔹 Predicting: {target}")
        print("="*60)

        X_train, X_val, X_test, y_train, y_val, y_test = data_tuple
        if y_train.nunique() <= 1:
            print(f"⚠️ Target '{target}' has <=1 unique value — skipped.")
            continue

        results = []

        for name, model in models.items():
            print(f"Training {name} ...")
            start = time.time()
            try:
                model.fit(X_train, y_train)
                preds_val = model.predict(X_val)
                mae = mean_absolute_error(y_val, preds_val)
                rmse = math.sqrt(mean_squared_error(y_val, preds_val))
                r2 = r2_score(y_val, preds_val)
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                mae = rmse = r2 = np.nan
                model = None

            results.append({
                "Target": target,
                "Model": name,
                "MAE_val": mae,
                "RMSE_val": rmse,
                "R2_val": r2,
                "Train Time (s)": round(time.time() - start, 2)
            })

        results_df = pd.DataFrame(results).sort_values(by="R2_val", ascending=False).reset_index(drop=True)
        display(results_df)

        # Best model
        best_row = results_df.dropna(subset=["R2_val"]).iloc[0]
        best_model_name = best_row["Model"]
        best_model = models[best_model_name]
        best_models_per_target[target] = best_model_name

        # Retrain on train+val
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        best_model.fit(X_full, y_full)

        # Test evaluation
        preds_test = best_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, preds_test)
        test_rmse = math.sqrt(mean_squared_error(y_test, preds_test))
        test_r2 = r2_score(y_test, preds_test)

        # Train & val metrics for best model
        preds_train = best_model.predict(X_train)
        train_mae = mean_absolute_error(y_train, preds_train)
        train_rmse = math.sqrt(mean_squared_error(y_train, preds_train))
        train_r2 = r2_score(y_train, preds_train)

        preds_val_full = best_model.predict(X_val)
        val_mae = mean_absolute_error(y_val, preds_val_full)
        val_rmse = math.sqrt(mean_squared_error(y_val, preds_val_full))
        val_r2 = r2_score(y_val, preds_val_full)

        print(f"\n✅ Test Metrics for {target} using {best_model_name}:")
        print(f"R²: {test_r2:.3f} | RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f}")

        _plot_pred_vs_residuals(y_test, preds_test, target)

        per_target_summaries.append({
            "Target": target,
            "Best Model": best_model_name,
            "Train R²": train_r2,
            "Train RMSE": train_rmse,
            "Train MAE": train_mae,
            "Val R²": val_r2,
            "Val RMSE": val_rmse,
            "Val MAE": val_mae,
            "Test R²": test_r2,
            "Test RMSE": test_rmse,
            "Test MAE": test_mae
        })

    full_summary_df = pd.DataFrame(per_target_summaries).sort_values(by="Test R²", ascending=False)
    display(full_summary_df)

    # ------------------------------
    # Summary Heatmaps: Train / Val / Test
    # ------------------------------
    metrics_r2 = full_summary_df[["Target","Train R²","Val R²","Test R²"]].set_index("Target")
    metrics_rmse = full_summary_df[["Target","Train RMSE","Val RMSE","Test RMSE"]].set_index("Target")

    # Sort by Test R² descending
    metrics_r2 = metrics_r2.sort_values(by="Test R²", ascending=False)
    metrics_rmse = metrics_rmse.loc[metrics_r2.index]

    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    sns.heatmap(metrics_r2, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("R² — Train / Val / Test")
    sns.heatmap(metrics_rmse, annot=True, fmt=".3f", cmap="magma", ax=axes[1])
    axes[1].set_title("RMSE — Train / Val / Test")
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Frequency of best model types
    # ------------------------------
    plt.figure(figsize=(7,5))
    model_counts = full_summary_df["Best Model"].value_counts()
    sns.barplot(x=model_counts.index, y=model_counts.values)
    plt.title("Frequency of Best Model Types")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return full_summary_df, best_models_per_target
