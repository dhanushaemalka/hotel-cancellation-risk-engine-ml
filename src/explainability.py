"""
Explainable AI (XAI) and Feature Attribution Module using SHAP.
Domain 6: Tourism & Hospitality - Hotel Booking Demand
IT3091 Machine Learning Group Project
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "assets")


def compute_shap_explanations(model, X_test: np.ndarray, feature_names: list, sample_size: int = 1500) -> dict:
    """
    Computes global and local SHAP explanations for tree-based models.
    Exports summary bar plots, beeswarm plots, and top feature rankings.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    print(f"[Explainability] Computing SHAP values on sample of {sample_size:,} records...")
    sample_indices = np.random.RandomState(42).choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    X_sample = X_test[sample_indices]
    
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle binary classification shap output formatting
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals_matrix = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_vals_matrix = shap_values[:, :, 1]
    else:
        shap_vals_matrix = shap_values
        
    # Global feature importance: mean absolute SHAP
    mean_abs_shap = np.mean(np.abs(shap_vals_matrix), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]
    
    top_features = [
        {
            "rank": int(i + 1),
            "feature": feature_names[idx],
            "importance": round(float(mean_abs_shap[idx]), 4)
        }
        for i, idx in enumerate(top_indices)
    ]
    
    # 1. Global Feature Importance Bar Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    top_feat_names = [feature_names[i] for i in top_indices[::-1]]
    top_shap_vals = [mean_abs_shap[i] for i in top_indices[::-1]]
    
    ax.barh(top_feat_names, top_shap_vals, color="#2563EB", edgecolor="black", height=0.7)
    ax.set_title("Global Feature Importance (Mean |SHAP Value| on Cancellation Risk)", fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Model Log-Odds Output)", fontweight="semibold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig12_shap_feature_importance.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig12_shap_feature_importance.png"), dpi=300)
    plt.close(fig)
    
    # 2. SHAP Summary Beeswarm Plot
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_vals_matrix, X_sample, feature_names=feature_names, max_display=15, show=False)
    plt.title("SHAP Beeswarm Plot: Feature Impact Direction on Cancellation Probability", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig13_shap_beeswarm.png"), dpi=300)
    plt.savefig(os.path.join(ASSETS_DIR, "fig13_shap_beeswarm.png"), dpi=300)
    plt.close()
    
    # 3. Local Explanation Extraction for Representative Archetypes
    # Case A: High Risk Booking
    high_risk_idx = np.argmax(model.predict_proba(X_sample)[:, 1])
    # Case B: Low Risk Booking
    low_risk_idx = np.argmin(model.predict_proba(X_sample)[:, 1])
    
    local_cases = {
        "high_risk_example": {
            "predicted_probability": round(float(model.predict_proba(X_sample[high_risk_idx:high_risk_idx+1])[:, 1][0]), 3),
            "top_positive_push": [
                {"feature": feature_names[i], "shap_value": round(float(shap_vals_matrix[high_risk_idx, i]), 3)}
                for i in np.argsort(shap_vals_matrix[high_risk_idx])[::-1][:5]
            ]
        },
        "low_risk_example": {
            "predicted_probability": round(float(model.predict_proba(X_sample[low_risk_idx:low_risk_idx+1])[:, 1][0]), 3),
            "top_negative_push": [
                {"feature": feature_names[i], "shap_value": round(float(shap_vals_matrix[low_risk_idx, i]), 3)}
                for i in np.argsort(shap_vals_matrix[low_risk_idx])[:5]
            ]
        }
    }
    
    # Save explainability artifacts
    with open(os.path.join(FIGURES_DIR, "shap_summary.json"), "w") as f:
        json.dump({"top_features": top_features, "local_cases": local_cases}, f, indent=2)
    with open(os.path.join(ASSETS_DIR, "shap_summary.json"), "w") as f:
        json.dump({"top_features": top_features, "local_cases": local_cases}, f, indent=2)
        
    print("[Explainability] SHAP computations and figures complete.")
    return {"top_features": top_features, "local_cases": local_cases}


if __name__ == "__main__":
    pass
