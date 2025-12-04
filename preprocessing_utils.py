# preprocessing_utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def preprocess_data(
    df,
    target_column,
    include_features=None,
    exclude_features=None,
    encoding="auto",
    scale=True,
    scaling_method="standard",
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    cardinality_threshold=50,
    drop_first=True,
    stratify=True  # ✅ NEW: enable stratification
):
    """
    Preprocesses cleaned data for model training with optional stratification.
    Stratification:
      - Classification → stratify by the target directly
      - Regression → stratify by quantile-binned target
    """

    # === 1. Select features ===
    X = df.drop(columns=[target_column])
    X = X.select_dtypes(exclude=["datetime64[ns]"])
    y = df[target_column]

    if include_features is not None:
        print(f"📋 Including only specified features: {include_features}")
        X = X[include_features]

    if exclude_features is not None:
        print(f"🚫 Excluding features: {exclude_features}")
        X = X.drop(columns=exclude_features, errors="ignore")

    # === 2. Encode categorical features ===
    cat_cols = X.select_dtypes(include="object").columns

    if encoding == "label":
        print("🔤 Using Label Encoding for all categorical features...")
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    elif encoding == "onehot":
        print("🏷️ Using One-Hot Encoding for all categorical features...")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=drop_first)

    elif encoding == "auto":
        print("🤖 Using Auto Encoding (adaptive to cardinality)...")
        small_cats, large_cats = [], []
        for col in cat_cols:
            unique_count = X[col].nunique()
            if unique_count <= cardinality_threshold:
                small_cats.append(col)
            else:
                large_cats.append(col)

        print(f"✅ One-Hot Encoding applied to: {small_cats}")
        print(f"✅ Label Encoding applied to: {large_cats}")

        if small_cats:
            X = pd.get_dummies(X, columns=small_cats, drop_first=drop_first)

        for col in large_cats:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    else:
        raise ValueError("encoding must be 'label', 'onehot', or 'auto'")

    # === 3. Scale numeric features ===
    if scale:
        num_cols = X.select_dtypes(include=["float64", "int64"]).columns
        scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()

        print(
            "⚖️ Applying StandardScaler (mean=0, std=1)..."
            if scaling_method == "standard"
            else "📏 Applying MinMaxScaler (range [0, 1])..."
        )
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # === 4. Stratification setup ===
    stratify_train = None

    if stratify:
        # Classification (discrete target)
        if y.nunique() <= 20:  
            stratify_train = y
            print("🔍 Using stratification for classification target.")
        else:
            # Regression → bin the continuous target
            try:
                stratify_train = pd.qcut(y, q=10, duplicates="drop")
                print("🔍 Using quantile-based stratification for regression target.")
            except:
                # fallback → no stratification
                stratify_train = None
                print("⚠️ Stratification skipped (could not bin target).")

    # === 5. Split into Train/Test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_train,
    )

    # === 6. Stratify again for Train/Val ===
    stratify_val = None
    if stratify and stratify_train is not None:
        try:
            stratify_val = pd.qcut(y_train, q=10, duplicates="drop") if y.nunique() > 20 else y_train
        except:
            stratify_val = None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_val
    )

    # === 7. Summary ===
    print("\n✅ Data successfully preprocessed and split:")
    print(f"  Training set:   {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set:       {X_test.shape[0]} samples")
    print(f"  Total features: {X.shape[1]} columns")

    return X_train, X_val, X_test, y_train, y_val, y_test
