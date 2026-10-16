"""
Unsupervised Customer & Booking Behavioral Segmentation Module.
Domain 6: Tourism & Hospitality - Hotel Booking Demand
IT3091 Machine Learning Group Project (Secondary Decision Lens)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "assets")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def perform_customer_segmentation(df_featured: pd.DataFrame, n_clusters: int = 4, sample_size: int = 15000):
    """
    Performs K-Means clustering on behavioral dimensions to discover actionable customer personas.
    Computes Elbow curve, Silhouette scores, and PCA 2D projections.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Behavioral feature subspace
    segment_features = [
        "lead_time",
        "total_stay",
        "total_guests",
        "adr",
        "cancellation_ratio",
        "total_of_special_requests",
        "required_car_parking_spaces",
        "is_repeated_guest"
    ]
    
    # Filter valid non-null rows
    X_seg = df_featured[segment_features].copy().dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)
    
    # 1. Elbow and Silhouette Analysis for K in [2..7] on sample
    print("[Segmentation] Computing Elbow Curve & Silhouette Scores...")
    sample_indices = np.random.RandomState(42).choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_indices]
    
    k_range = range(2, 8)
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_k.fit(X_sample)
        inertias.append(float(kmeans_k.inertia_))
        sil_score = silhouette_score(X_sample, kmeans_k.labels_, sample_size=3000, random_state=42)
        silhouettes.append(float(sil_score))
        
    # Plot Elbow & Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(list(k_range), inertias, marker="o", color="#2563EB", linewidth=2)
    axes[0].axvline(x=n_clusters, color="#DC2626", linestyle="--", label=f"Selected K={n_clusters}")
    axes[0].set_title("Elbow Method: Inertia vs Cluster Count", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    axes[0].legend()
    
    axes[1].plot(list(k_range), silhouettes, marker="s", color="#059669", linewidth=2)
    axes[1].axvline(x=n_clusters, color="#DC2626", linestyle="--", label=f"Selected K={n_clusters}")
    axes[1].set_title("Silhouette Coefficient vs Cluster Count", fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_cluster_elbow_silhouette.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig7_cluster_elbow_silhouette.png"), dpi=300)
    plt.close(fig)
    
    # 2. Fit final K-Means with K=4
    print(f"[Segmentation] Fitting Final K-Means with K={n_clusters}...")
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    X_seg["cluster"] = cluster_labels
    X_seg["is_canceled"] = df_featured.loc[X_seg.index, "is_canceled"]
    
    # Persona Naming & Profiling
    cluster_profiles = {}
    persona_names = {
        0: "Spontaneous Short-Stay Business",
        1: "High-Risk Advance Long-Lead Planner",
        2: "Leisure Family / High-Value Vacation",
        3: "Loyal Repeated / Direct Guest"
    }
    
    # Calculate means per cluster
    cluster_summary = X_seg.groupby("cluster").mean()
    cluster_counts = X_seg["cluster"].value_counts().sort_index()
    
    # Reassign intuitive personas based on lead_time, adr, cancellation rate
    # Let's inspect cluster statistics dynamically
    persona_mapping = {}
    for c in range(n_clusters):
        c_stats = cluster_summary.loc[c].to_dict()
        lead = c_stats["lead_time"]
        cancel = c_stats["is_canceled"]
        guests = c_stats["total_guests"]
        repeated = c_stats["is_repeated_guest"]
        
        if repeated > 0.3:
            name = "Loyal Repeated / Low-Risk Direct"
            action = "Offer VIP loyalty perks, zero-deposit guarantee, and express check-in."
        elif guests > 2.2:
            name = "Family & Multi-Guest Vacationer"
            action = "Promote family dining packages, confirm child bedding, upsell parking."
        elif lead > 120 or cancel > 0.45:
            name = "High-Risk Long-Lead Booker"
            action = "Enforce 50% non-refundable deposit, send automated 14-day re-confirmation nudge."
        else:
            name = "Spontaneous Short-Stay Transient"
            action = "Flexible cancellation, mobile keyless entry, last-minute room upgrades."
            
        persona_mapping[c] = {
            "persona_name": name,
            "count": int(cluster_counts[c]),
            "pct": round(float(cluster_counts[c]) / len(X_seg) * 100, 1),
            "lead_time_mean": round(c_stats["lead_time"], 1),
            "stay_nights_mean": round(c_stats["total_stay"], 1),
            "adr_mean": round(c_stats["adr"], 2),
            "total_guests_mean": round(c_stats["total_guests"], 2),
            "cancellation_rate_pct": round(c_stats["is_canceled"] * 100, 1),
            "special_requests_mean": round(c_stats["total_of_special_requests"], 2),
            "recommended_strategy": action
        }
    
    # Save Cluster Model and Scaler
    joblib.dump(kmeans_final, os.path.join(PROCESSED_DIR, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "kmeans_scaler.joblib"))
    
    with open(os.path.join(PROCESSED_DIR, "customer_personas.json"), "w") as f:
        json.dump(persona_mapping, f, indent=2)
    with open(os.path.join(ASSETS_DIR, "customer_personas.json"), "w") as f:
        json.dump(persona_mapping, f, indent=2)

    # 3. PCA 2D Visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled[sample_indices])
    pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
    pca_df["cluster"] = [f"Cluster {c}: {persona_mapping[c]['persona_name']}" for c in cluster_labels[sample_indices]]
    
    fig, ax = plt.subplots(figsize=(11, 7))
    palette = ["#2563EB", "#DC2626", "#059669", "#D97706"]
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette=palette, alpha=0.6, s=25, ax=ax)
    ax.set_title(f"Customer Persona Clusters in 2D Principal Component Space\n(Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontweight="bold")
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.legend(title="Customer Archetype", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig8_customer_clusters_pca.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig8_customer_clusters_pca.png"), dpi=300)
    plt.close(fig)
    
    print("[Segmentation] Persona clustering complete and visualizations generated.")
    return persona_mapping


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import clean_raw_data, engineer_features
    df = load_raw_data()
    df_clean = clean_raw_data(df)
    df_feat = engineer_features(df_clean)
    perform_customer_segmentation(df_feat)
