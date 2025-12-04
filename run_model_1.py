# run_model_1.py

import time
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from preprocessing_utils import preprocess_data


def run_model_1(df_clean):

    print("🏗️ Running Model 1: Predicting 'CCME_Values' ...\n")

    # === 1️⃣ Preprocess ===
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df_clean,
        target_column="CCME_Values",
        encoding="auto",
        scale=True,
        scaling_method="standard",
        drop_first=False,
        random_state=42,
        exclude_features=["CCME_WQI", "Date_String"]
    )

    feature_list = X_train.columns.tolist()
    print("\n📋 Feature and Target Summary")
    print(f"Target: CCME_Values")
    print(f"Features ({len(feature_list)} total): {', '.join(feature_list)}")

    # === 2️⃣ Models ===
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbosity=0),
        "MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }

    # === 3️⃣ Train ALL Models on Train Set ===
    results = []

    for name, model in models.items():
        print(f"\n🔹 Training {name} ...")
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()

        # --- TRAIN METRICS ---
        train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)

        # --- VALIDATION METRICS ---
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)

        # --- GENERALIZATION GAP ---
        gap = abs(val_rmse - train_rmse)

        results.append({
            "Model": name,
            "Train RMSE": train_rmse,
            "Val RMSE": val_rmse,
            "Gap (RMSE)": gap,
            "Train MAE": train_mae,
            "Val MAE": val_mae,
            "Train R²": train_r2,
            "Val R²": val_r2,
            "Train Time (s)": round(end - start, 2)
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # === 4️⃣ ACCURACY-FIRST SELECTION ===
    print("\n🔎 Selecting best model (lowest Validation RMSE + overfitting check)...")

    median_gap = results_df["Gap (RMSE)"].median()
    max_allowed_gap = 2 * median_gap    # threshold

    eligible = results_df[results_df["Gap (RMSE)"] <= max_allowed_gap]

    if eligible.empty:
        print("⚠ WARNING: All models overfit. Selecting lowest Val RMSE anyway.")
        eligible = results_df

    # Select best model based on lowest Validation RMSE
    best_row = eligible.loc[eligible["Val RMSE"].idxmin()]
    best_model_name = best_row["Model"]

    print(f"\n🏆 Best Model (Accuracy-First): {best_model_name}")
    print(f"   Validation RMSE = {best_row['Val RMSE']:.4f}")
    print(f"   Gap (RMSE)     = {best_row['Gap (RMSE)']:.4f}")

    # === 5️⃣ Retrain Best Model on Train + Validation ===
    best_model = models[best_model_name]

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    best_model.fit(X_train_full, y_train_full)

    # ======================================================================
    # 🚨 STOP HERE DURING HYPERPARAMETER TUNING (DO NOT TOUCH THE TEST SET)
    # ======================================================================

    # === 6️⃣ Evaluate on Test Set (DISABLED TO AVOID DATA LEAKAGE) ===
    # ❌ DO NOT UNCOMMENT THESE LINES UNTIL ALL TUNING & MODEL SELECTION IS DONE
    #
    # test_pred = best_model.predict(X_test)
    # test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    # test_mae = mean_absolute_error(y_test, test_pred)
    # test_r2 = r2_score(y_test, test_pred)
    #
    # print(f"\n📌 Final TEST Performance (Best Model: {best_model_name})")
    # print(f"   Test RMSE = {test_rmse:.4f}")
    # print(f"   Test MAE  = {test_mae:.4f}")
    # print(f"   Test R²   = {test_r2:.4f}")

    # === 7️⃣ PRINT RESULTS TABLE (sorted by Val RMSE) ===
    results_df = results_df.sort_values("Val RMSE").reset_index(drop=True)

    # print("\n📊 Train/Validation Results Table:")
    # print(results_df.to_string(index=False))

    return results_df, best_model, X_train, y_train, X_val, y_val, X_test, y_test





