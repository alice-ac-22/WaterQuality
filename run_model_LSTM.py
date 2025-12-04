# run_model_lstm.py

import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def create_sliding_windows(df, target_column="CCME_Values", window_size=30):
    """
    Creates sliding windows across the full dataset:
        X shape -> (num_sequences, window_size, num_features)
        y shape -> (num_sequences,)
    """

    print(f"\n📌 Creating sliding windows (window size = {window_size}) ...")

    # --- Sort by date to preserve temporal order ---
    df = df.sort_values("Date_Ordinal").reset_index(drop=True)

    # --- Select features ---
    features = df.drop(columns=[target_column]).values
    target = df[target_column].values

    X, y = [], []

    for i in range(len(df) - window_size):
        X.append(features[i:i+window_size])
        y.append(target[i+window_size])

    X = np.array(X)
    y = np.array(y)

    print(f"   → Created {X.shape[0]:,} sequences")
    print(f"   → Input shape: {X.shape}")

    return X, y, df.drop(columns=[target_column]).columns.tolist()


def train_val_test_split_sequences(X, y, train_ratio=0.7, val_ratio=0.1):
    """
    Split sequential windows into Train / Val / Test.
    """

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\n📊 Sequence Split:")
    print(f"   Train: {len(X_train):,}")
    print(f"   Val:   {len(X_val):,}")
    print(f"   Test:  {len(X_test):,}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_lstm_model(input_shape):
    """
    Builds a simple LSTM regression model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    return model


def run_model_lstm(df_clean, window_size=30):
    print("\n🏗️ Running LSTM Model: Predicting 'CCME_Values' using Sequential Windows ...")

    # === 1️⃣ Scale ALL numeric features ===
    print("\n⚖️ Scaling all features except target...")
    scaler = StandardScaler()

    feature_cols = df_clean.drop(columns=["CCME_Values"]).columns
    df_scaled = df_clean.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

    # === 2️⃣ Create sliding windows ===
    X, y, feature_list = create_sliding_windows(
        df_scaled,
        target_column="CCME_Values",
        window_size=window_size
    )

    # === 3️⃣ Train/Val/Test split ===
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_sequences(X, y)

    # === 4️⃣ Build model ===
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    print("\n🧠 Training LSTM model...")
    start = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        verbose=1
    )

    end = time.time()

    # === 5️⃣ Evaluate Train/Validation ===
    print("\n📈 Evaluating model...")

    # TRAIN
    train_pred = model.predict(X_train).flatten()
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    # VAL
    val_pred = model.predict(X_val).flatten()
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)

    print(f"\n🔹 Train RMSE: {train_rmse:.4f}")
    print(f"🔹 Train MAE:  {train_mae:.4f}")
    print(f"🔹 Train R²:   {train_r2:.4f}")

    print(f"\n🔹 Val RMSE:   {val_rmse:.4f}")
    print(f"🔹 Val MAE:    {val_mae:.4f}")
    print(f"🔹 Val R²:     {val_r2:.4f}")

    print(f"\n⏱️ Total Training Time: {end - start:.2f} seconds")

    # === 6️⃣ Return consistent format ===
    results = {
        "Train RMSE": train_rmse,
        "Val RMSE": val_rmse,
        "Train MAE": train_mae,
        "Val MAE": val_mae,
        "Train R²": train_r2,
        "Val R²": val_r2,
        "Train Time (s)": round(end - start, 2)
    }

    return results, model, X_train, y_train, X_val, y_val, X_test, y_test