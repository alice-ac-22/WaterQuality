# cleaning_rawdata.py

import os
import pandas as pd
import numpy as np

def clean_raw_data(df):
    """
    Clean raw dataset by:
      1. Handling missing values (mean for numeric, mode for categorical).
      2. Converting 'Date' column to datetime format.
      3. Adding time-based features (Year, Month, Day).
      4. Creating cyclical encodings (sin/cos) for LSTM or RNN models.
      5. Keeping both linear (Date_Ordinal) and cyclical (sin/cos) time features.
      6. Generating a readable Date_String for grouping or export.

    Returns:
        pd.DataFrame: Cleaned dataset with new time features.
    """

    # === 1. Handle missing numeric columns by replacing NaN with column mean ===
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        mean_val = df[col].mean()                          # calculate mean of numeric column
        df[col] = df[col].fillna(mean_val)                 # replace missing values with mean

    # === 2. Handle missing categorical columns by replacing NaN with most frequent (mode) value ===
    for col in df.select_dtypes(include=['object']).columns:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else None
        df[col] = df[col].fillna(mode_val)                 # replace missing strings with mode

    # === 3. Convert 'Date' to datetime format (DD-MM-YYYY) ===
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

    # === 4. Extract simple time components ===
    df['Year']  = df['Date'].dt.year                       # extract year
    df['Month'] = df['Date'].dt.month                      # extract month
    df['Day']   = df['Date'].dt.day                        # extract day

    # === 5. Create linear and cyclical encodings ===
    # Linear ordinal (numeric timeline)
    df['Date_Ordinal'] = df['Date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)

    # Cyclical features to capture seasonality
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)

    # Normalized year (0–1 range for long-term trend modeling)
    df['Year_norm'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min())

    # === 6. Add human-readable date string (for grouping or export) ===
    df['Date_String'] = df['Date'].dt.strftime('%d-%m-%Y')

    return df
