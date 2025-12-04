# run_model_2_softmax_regression.py
"""
Model 2:
Predict 'CCME_WQI' (categorical) using Softmax (Multinomial Logistic Regression).
This module preprocesses data, trains the model, evaluates performance,
plots confusion matrix and other diagnostics, and returns results.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

from preprocessing_utils import preprocess_data


def run_softmax_model(df_clean):

    print("🏗️ Running Model 2: Predicting 'CCME_WQI' ...\n")

    # === 1️⃣ Preprocess ===
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df_clean,
        target_column="CCME_WQI",
        encoding="auto",
        scale=True,
        scaling_method="standard",
        drop_first=False,
        random_state=42,
        exclude_features=["CCME_Values", "Date_String"]
    )

    feature_list = X_train.columns.tolist()
    print("\n📋 Feature Summary")
    print(f"Features ({len(feature_list)}): {', '.join(feature_list)}")

    # === 2️⃣ Encode labels ===
    target_order = ["Excellent", "Good", "Fair", "Marginal", "Poor"]
    le = LabelEncoder()
    le.fit(target_order)

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # === 3️⃣ Softmax Model ===
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )

    # === 4️⃣ Train ===
    print("\n🔹 Training Softmax Regression ...")
    start = time.time()
    model.fit(X_train, y_train_enc)
    end = time.time()

    # === NEW: TRAINING METRICS ===
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train_enc, train_preds)
    train_precision = precision_score(y_train_enc, train_preds, average="weighted")
    train_recall = recall_score(y_train_enc, train_preds, average="weighted")
    train_f1_weighted = f1_score(y_train_enc, train_preds, average="weighted")
    train_f1_macro = f1_score(y_train_enc, train_preds, average="macro")

    print("\n📘 Training Metrics:")
    print(f"Accuracy:  {train_acc:.3f}")
    print(f"Precision (weighted): {train_precision:.3f}")
    print(f"Recall (weighted):    {train_recall:.3f}")
    print(f"F1 (weighted):        {train_f1_weighted:.3f}")
    print(f"F1 (macro):           {train_f1_macro:.3f}")

    # === VALIDATION METRICS ===
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val_enc, val_preds)
    val_precision = precision_score(y_val_enc, val_preds, average="weighted")
    val_recall = recall_score(y_val_enc, val_preds, average="weighted")
    val_f1_weighted = f1_score(y_val_enc, val_preds, average="weighted")
    val_f1_macro = f1_score(y_val_enc, val_preds, average="macro")

    print("\n📙 Validation Metrics:")
    print(f"Accuracy:  {val_acc:.3f}")
    print(f"Precision (weighted): {val_precision:.3f}")
    print(f"Recall (weighted):    {val_recall:.3f}")
    print(f"F1 (weighted):        {val_f1_weighted:.3f}")
    print(f"F1 (macro):           {val_f1_macro:.3f}")

    # === 5️⃣ Retrain on full train+val ===
    X_train_full = pd.concat([X_train, X_val])
    y_train_full_enc = np.concatenate([y_train_enc, y_val_enc])
    model.fit(X_train_full, y_train_full_enc)

    # === 6️⃣ Test Metrics ===
    test_preds = model.predict(X_test)

    test_acc = accuracy_score(y_test_enc, test_preds)
    test_precision = precision_score(y_test_enc, test_preds, average="weighted")
    test_recall = recall_score(y_test_enc, test_preds, average="weighted")
    test_f1_weighted = f1_score(y_test_enc, test_preds, average="weighted")
    test_f1_macro = f1_score(y_test_enc, test_preds, average="macro")

    print(f"\n🏆 Test Accuracy: {test_acc:.3f}")
    print(f"🏆 Test Precision (weighted): {test_precision:.3f}")
    print(f"🏆 Test Recall (weighted):    {test_recall:.3f}")
    print(f"🏆 Test F1 (weighted):        {test_f1_weighted:.3f}")
    print(f"🏆 Test F1 (macro):           {test_f1_macro:.3f}")

    # === 6.1️⃣ Regression-style metrics ===
    test_mae = mean_absolute_error(y_test_enc, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test_enc, test_preds))
    test_r2 = r2_score(y_test_enc, test_preds)

    print("\n📊 Additional Test Metrics (numeric labels)")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R²:   {test_r2:.4f}")

    # === 7️⃣ Confusion Matrix ===
    cm = confusion_matrix(y_test_enc, test_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Softmax Regression")
    plt.show()

    # === 8️⃣ Classification Report ===
    print("\n📄 Classification Report:\n")
    print(classification_report(y_test_enc, test_preds, target_names=le.classes_))

    # === 9️⃣ True vs Predicted Plot ===
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_enc[:100], label="True", marker="o", linestyle="--")
    plt.plot(test_preds[:100], label="Predicted", marker="x", linestyle=":")
    plt.title("True vs Predicted Labels (First 100 Samples)")
    plt.legend()
    plt.show()

    # === 🔟 Summary Table ===
    results_df = pd.DataFrame([{
        "Model": "Softmax Regression",

        "Train Accuracy": train_acc,
        "Train Precision": train_precision,
        "Train Recall": train_recall,
        "Train F1 Weighted": train_f1_weighted,
        "Train F1 Macro": train_f1_macro,

        "Validation Accuracy": val_acc,
        "Validation Precision": val_precision,
        "Validation Recall": val_recall,
        "Validation F1 Weighted": val_f1_weighted,
        "Validation F1 Macro": val_f1_macro,

        "Test Accuracy": test_acc,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1 Weighted": test_f1_weighted,
        "Test F1 Macro": test_f1_macro,

        "Test MAE": test_mae,
        "Test RMSE": test_rmse,
        "Test R2": test_r2,

        "Train Time (s)": round(end - start, 2)
    }])

    print("\n📊 Summary Table:")
    print(results_df.to_string(index=False))

    return results_df, model
