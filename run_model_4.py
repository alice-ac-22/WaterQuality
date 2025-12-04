# ===========================
# run_model_4.py (Chemistry-only PCA)
# ===========================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run_model_4(df_clean):

    print("🔍 Running Unsupervised Model 4 (Chemistry-only PCA)...\n")

    # ==========================================================
    # 1️⃣ SELECT ONLY RAW CHEMISTRY FEATURES
    # ==========================================================
    chemistry_features = [
        "Ammonia (mg/l)",
        "Biochemical Oxygen Demand (mg/l)",
        "Dissolved Oxygen (mg/l)",
        "Orthophosphate (mg/l)",
        "pH (ph units)",
        "Temperature (cel)",
        "Nitrogen (mg/l)",
        "Nitrate (mg/l)"
    ]

    # Subset dataset
    df_chem = df_clean[chemistry_features].copy()

    # Drop missing rows
    df_chem = df_chem.dropna()
    print(f"📌 Using {df_chem.shape[0]} samples × {df_chem.shape[1]} chemistry features\n")

    # ==========================================================
    # 2️⃣ SCALE DATA
    # ==========================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_chem)

    # ==========================================================
    # 3️⃣ PCA (2 COMPONENTS)
    # ==========================================================
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("📉 PCA Explained Variance Ratio:")
    print(f"   PC1 = {pca.explained_variance_ratio_[0]:.3f}")
    print(f"   PC2 = {pca.explained_variance_ratio_[1]:.3f}\n")

    # PCA loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1_Loading", "PC2_Loading"],
        index=chemistry_features
    )

    print("🔎 Top contributors to PC1:")
    print(loadings["PC1_Loading"].abs().sort_values(ascending=False), "\n")

    print("🔎 Top contributors to PC2:")
    print(loadings["PC2_Loading"].abs().sort_values(ascending=False), "\n")

    # ==========================================================
    # 4️⃣ K-MEANS ON PCA SPACE
    # ==========================================================
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    silhouette = silhouette_score(X_pca, clusters)
    print(f"📊 Silhouette Score (k=2): {silhouette:.3f}\n")

    print("Cluster counts:")
    print(pd.Series(clusters).value_counts(), "\n")

    # ==========================================================
    # RETURN EVERYTHING
    # ==========================================================
    results = {
        "X_chem": df_chem,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "pca_model": pca,
        "pca_loadings": loadings,
        "clusters": clusters,
        "kmeans_model": kmeans,
        "silhouette_score": silhouette
    }

    print("✅ Chemistry-only Model 4 Completed.\n")

    return results
