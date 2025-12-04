#!/usr/bin/env python
# coding: utf-8

# ==========================
# Water Potability Visualization, Feature Correlation & Interpretation
# ==========================
# Author: ChatGPT for Reem Chakik
# Description:
#  - Visualizes key relationships between water quality parameters
#  - Compares variations by Waterbody Type and Country
#  - Identifies top correlated features with CCME_Value
#  - Exports feature correlations to Excel
#  - Prints interpretive summaries of findings

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import warnings

# --------------------------
# Suppress warnings for clean output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ==========================
# Functions to call from MAIN.ipynb
# ==========================

def run_displayed_visualizations(output_folder="main_visualize"):
    """
    Displays only the first two visualizations:
    1. Correlation heatmap
    2. Feature correlation with CCME_Values
    Saves these plots and the correlation Excel summary in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load dataset
    df = pd.read_csv("Combined_dataset.csv")
    df = df.dropna(subset=["CCME_Values"])

    # Numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()

    # 1️⃣ Correlation Heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"), dpi=300)
    plt.show()  # ✅ Display

    # 2️⃣ Feature Correlation with CCME_Values
    if "CCME_Values" in corr.columns:
        ccme_corr = corr["CCME_Values"].sort_values(ascending=False)
        features = ccme_corr.drop("CCME_Values")

        plt.figure(figsize=(8, 5))
        plt.bar(features.index, features.values, color='skyblue')
        plt.title("Feature Correlation with CCME_Values")
        plt.ylabel("Correlation Coefficient")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "Feature_Correlation_with_CCME_Values.png"), dpi=300)
        plt.show()  # ✅ Display

        # Save correlation summary
        corr_df = ccme_corr.reset_index()
        corr_df.columns = ["Feature", "Correlation_with_CCME_Values"]
        corr_df.to_excel(os.path.join(output_folder, "feature_correlation_summary.xlsx"), index=False)

    #print(f"✅ Displayed visualizations completed. Files saved in '{output_folder}'.")


def run_silent_visualizations(output_folder="main_visualize/silent"):
    """
    Runs all other plots (boxplots, country-level, scatterplots, etc.)
    but does NOT display them; saves all figures to output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv("Combined_dataset.csv")
    df = df.dropna(subset=["CCME_Values"])
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # 7️⃣ Boxplots by Waterbody Type
    if 'Waterbody Type' in df.columns:
        for col in numeric_cols:
            if col != 'CCME_Values':
                plt.figure(figsize=(8,5))
                sns.boxplot(data=df, x='Waterbody Type', y=col)
                plt.title(f"{col} Variation by Waterbody Type")
                plt.xticks(rotation=45)
                plt.tight_layout()
                safe_col_name = re.sub(r'[\\/*?:"<>|()\s]+', '_', col)
                plt.savefig(os.path.join(output_folder, f"{safe_col_name}_by_WaterbodyType.png"), dpi=300)
                plt.close()  # ✅ Close figure to avoid display

    # 8️⃣ Country-Level Analysis
    if 'Country' in df.columns:
        key_vars = ["CCME_Value", "pH", "Orthophosphate (mg/l)",
                    "Dissolved Oxygen (mg/l)", "Ammonia (mg/l)"]
        for var in key_vars:
            if var in df.columns:
                plt.figure(figsize=(10,5))
                sns.boxplot(data=df, x='Country', y=var, palette='coolwarm')
                plt.title(f"{var} Variation by Country")
                plt.xticks(rotation=45)
                plt.tight_layout()
                safe_var_name = re.sub(r'[\\/*?:"<>|()\s]+', '_', var)
                plt.savefig(os.path.join(output_folder, f"{safe_var_name}_by_Country.png"), dpi=300)
                plt.close()

        # Average CCME_Value per Country
        country_mean = df.groupby('Country')["CCME_Values"].mean().sort_values(ascending=False)
        plt.figure(figsize=(10,5))
        sns.barplot(x=country_mean.index, y=country_mean.values, palette='viridis')
        plt.title("Average CCME_Values per Country")
        plt.ylabel("Mean CCME_Values")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "Average_CCME_Value_per_Country.png"), dpi=300)
        plt.close()

    # 9️⃣ Pairwise Scatterplots
    pairs = [
        ("pH", "Dissolved Oxygen (mg/l)"),
        ("Ammonia (mg/l)", "pH"),
        ("Ammonia (mg/l)", "Dissolved Oxygen (mg/l)"),
        ("Orthophosphate (mg/l)", "CCME_Values"),
        ("Biochemical Oxygen Demand (mg/l)", "CCME_Values"),
    ]
    for x, y in pairs:
        if x in df.columns and y in df.columns:
            plt.figure(figsize=(6,5))
            sns.scatterplot(data=df, x=x, y=y, hue="Country", palette="tab10", alpha=0.7)
            plt.title(f"{y} vs {x}")
            plt.tight_layout()
            safe_x = re.sub(r'[\\/*?:"<>|()\s]+', '_', x)
            safe_y = re.sub(r'[\\/*?:"<>|()\s]+', '_', y)
            plt.savefig(os.path.join(output_folder, f"{safe_y}_vs_{safe_x}.png"), dpi=300)
            plt.close()

    #print(f"✅ Silent visualizations saved in '{output_folder}'.")


# 🧠 What this Excel file shows
# 
# Each feature (e.g., pH, Ammonia, Nitrogen, etc.) is correlated with CCME_Values, which represents your overall water quality index.
# 
# The Correlation Coefficient values range from –1 to +1:
# 
# +1 → Perfect positive relationship
# 
# 0 → No linear relationship
# 
# –1 → Perfect negative relationship
# 
# 📊 Interpretation of your results
# Feature	Correlation	Interpretation
# CCME_Values	1.000	Reference (self-correlation).
# pH (ph units)	+0.08	Very weak positive correlation. pH changes have almost no clear effect on CCME score.
# Dissolved Oxygen (mg/l)	–0.03	Almost no relationship — slightly negative, but negligible.
# Temperature (cel)	–0.11	Weak negative correlation — as temperature rises, CCME may slightly decrease (warmer water can reduce oxygen solubility).
# Nitrate (mg/l)	–0.31	Moderate negative correlation — higher nitrate tends to reduce water quality.
# Biochemical Oxygen Demand (mg/l)	–0.32	Moderate negative correlation — higher BOD means more organic pollution → lower CCME.
# Ammonia (mg/l)	–0.39	Clearer negative correlation — higher ammonia indicates contamination → lower CCME.
# Nitrogen (mg/l)	–0.61	Strong negative correlation — high nitrogen pollution (e.g., from fertilizers or sewage) → much poorer water quality.
# Orthophosphate (mg/l)	–0.66	Strongest negative correlation — phosphorus compounds (orthophosphate) strongly lower CCME, likely due to eutrophication effects.
# ⚙️ Why some correlations are negative
# 
# Negative correlations mean that as the concentration of that parameter increases, the overall water quality (CCME Value) decreases.
# 
# In environmental terms:
# 
# Ammonia, Nitrate, Nitrogen, Orthophosphate, and BOD are pollutants.
# 
# So, higher pollutant levels → poorer water quality → lower CCME Values, which is reflected as a negative correlation.
# 
# 🧩 In summary
# 
# Parameters with strong negative correlation (like Orthophosphate and Nitrogen) are key drivers of water quality degradation in your dataset.
# 
# Parameters with weak or near-zero correlation (like pH or DO) are less influential in determining the CCME score in your current dataset.
# 
# You can visualize this insight as a bar chart (which your script already generated) — this visually highlights which parameters matter most.
# 
# Interpretation of Feature Correlation with CCME Values:
# The correlation analysis revealed that nutrient-related parameters, particularly Orthophosphate (r = –0.66) and Nitrogen (r = –0.61), have the strongest negative relationships with the CCME water quality index, indicating that higher concentrations of these pollutants substantially degrade water quality. Moderate negative correlations were also observed for Ammonia, Biochemical Oxygen Demand (BOD), and Nitrate, suggesting their notable contribution to water contamination. In contrast, parameters such as pH, Dissolved Oxygen, and Temperature showed weak or negligible correlations, implying limited influence on overall CCME scores in this dataset. Overall, the results highlight that nutrient enrichment and organic pollution are key factors deteriorating water quality across the sampled sites.
