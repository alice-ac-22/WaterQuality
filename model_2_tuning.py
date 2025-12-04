# model_2_tuning.py
"""
Softmax Regression (SGD-based) with:
- Hyperparameter tuning
- Full metrics for Train, Validation, and Test
- No plots displayed
"""

import time
import pandas as pd
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from scipy.stats import loguniform, randint
from preprocessing_utils import preprocess_data


# =====================================================
# Main Function
# =====================================================
def run_softmax_model(df_clean, tuning_fraction=0.10, n_iter_search=35, cv_folds=3):
    print("🏗️ Running Tuned Softmax (SGD) with Full Evaluation ...\n")

    # === Preprocessing ================================================
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df_clean,
        target_column="CCME_WQI",
        encoding="auto",
        scale=False,
        scaling_method="standard",
        drop_first=False,
        random_state=42,
        exclude_features=["CCME_Values", "Date_String"]
    )

    # Train + validation merged
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    # === Label Encoding ===============================================
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_val, y_test]).astype(str))

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)
    print(f"Detected classes: {list(le.classes_)}\n")

    # === Subset for tuning ============================================
    desired_n = max(1, int(len(X_train) * tuning_fraction))
    min_needed = max(n_classes * 3, cv_folds * n_classes)
    if desired_n < min_needed:
        desired_n = min_needed

    X_tune, _, y_tune, _ = train_test_split(
        X_train,
        y_train,
        train_size=desired_n,
        stratify=y_train,
        random_state=42
    )
    y_tune_enc = le.transform(y_tune)
    print(f"🧮 Using {len(X_tune)} samples (~{tuning_fraction*100:.1f}%) for tuning.\n")

    # === SGD Softmax Pipeline =========================================
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            random_state=42,
            shuffle=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            tol=None
        ))
    ])

    # === Hyperparameter Space ==========================================
    param_distributions = {
        "clf__alpha": loguniform(1e-7, 1e-2),
        "clf__eta0": loguniform(1e-3, 1e-1),
        "clf__learning_rate": ["constant", "optimal", "adaptive"],
        "clf__penalty": ["l2", "elasticnet"],
        "clf__l1_ratio": [0.0, 0.3, 0.5],
        "clf__max_iter": randint(300, 2000),
        "clf__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {"f1_weighted": "f1_weighted", "accuracy": "accuracy"}

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        scoring=scoring,
        refit="f1_weighted",
        cv=cv,
        random_state=42,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )

    print("🔍 Performing hyperparameter search...\n")
    t0 = time.time()
    search.fit(X_tune, y_tune_enc)
    t1 = time.time()

    print(f"\n✅ Tuning completed in {t1 - t0:.2f} sec")
    print("🏆 Best Params:")
    for k, v in search.best_params_.items():
        print(f"   {k}: {v}")

    # === Train Best Model Fully =======================================
    best_model = search.best_estimator_
    print("\n🔹 Training best model on full training set...")
    t0_final = time.time()
    best_model.fit(X_train_full, le.transform(y_train_full))
    t1_final = time.time()

    # =====================================================
    # Compute Metrics for Train / Validation / Test
    # =====================================================
    def compute_metrics(x, y_true, dataset_name):
        y_pred = best_model.predict(x)
        return {
            "dataset": dataset_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred)
        }

    train_metrics = compute_metrics(X_train, y_train_enc, "train")
    val_metrics = compute_metrics(X_val, y_val_enc, "validation")
    test_metrics = compute_metrics(X_test, y_test_enc, "test")

    # === Summary Table ================================================
    summary = pd.DataFrame([train_metrics, val_metrics, test_metrics])

    # Print metrics
    print("\n📊 Train / Validation / Test Metrics:")
    print(summary)

    return summary, best_model, search.cv_results_
