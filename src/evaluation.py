"""
Comprehensive Model Evaluation, Probability Calibration, and Cost-Benefit Revenue Optimization.
Domain 6: Tourism & Hospitality - Hotel Booking Demand
IT3091 Machine Learning Group Project
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "assets")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def evaluate_models_on_test_set(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluates all candidate models on the held-out test set across standard ML metrics and Brier score.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    test_metrics = {}
    model_predictions = {}
    
    print("[Evaluation] Evaluating candidate models on Test Set (N = {:,})...".format(len(y_test)))
    print("=" * 80)
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict(X_test)
            
        preds_default = (probs >= 0.5).astype(int) if name not in ["Dummy (Majority Class)", "Heuristic (Lead-Time Rule)"] else model.predict(X_test)
        
        # Compute metrics
        acc = accuracy_score(y_test, preds_default)
        prec = precision_score(y_test, preds_default, zero_division=0)
        rec = recall_score(y_test, preds_default, zero_division=0)
        f1 = f1_score(y_test, preds_default, zero_division=0)
        
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc_val = auc(fpr, tpr)
        
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, probs)
        pr_auc_val = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        
        test_metrics[name] = {
            "model_name": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc_val), 4),
            "pr_auc": round(float(pr_auc_val), 4),
            "brier_score": round(float(brier), 4)
        }
        
        model_predictions[name] = {
            "probs": probs,
            "preds": preds_default,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "pr_precision": pr_precision.tolist(),
            "pr_recall": pr_recall.tolist()
        }
        
        print(f"{name:<30} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc_val:.4f} | PR-AUC: {pr_auc_val:.4f} | Brier: {brier:.4f}")
        
    print("=" * 80)
    
    # Save test metrics JSON
    with open(os.path.join(FIGURES_DIR, "test_metrics_comparison.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(ASSETS_DIR, "test_metrics_comparison.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
        
    return test_metrics, model_predictions


def plot_roc_and_pr_curves(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray):
    """
    Plots multi-model ROC Curves and Precision-Recall Curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Color palette
    colors = {
        "Dummy (Majority Class)": "#9CA3AF",
        "Heuristic (Lead-Time Rule)": "#F59E0B",
        "Logistic Regression": "#3B82F6",
        "Random Forest": "#10B981",
        "XGBoost": "#8B5CF6",
        "LightGBM": "#EC4899",
        "Multi-Layer Perceptron (MLP)": "#06B6D4"
    }
    
    # 1. ROC Curves
    for name, model in models_dict.items():
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.3f})", color=colors.get(name, "#333333"), linewidth=2 if "XGBoost" in name or "LightGBM" in name else 1.5)
        
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.500)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Comparison", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontweight="semibold")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="semibold")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. Precision-Recall Curves
    baseline_pr = float(y_test.mean())
    for name, model in models_dict.items():
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap_val = average_precision_score(y_test, probs)
        axes[1].plot(rec, prec, label=f"{name} (AP = {ap_val:.3f})", color=colors.get(name, "#333333"), linewidth=2 if "XGBoost" in name or "LightGBM" in name else 1.5)
        
    axes[1].axhline(y=baseline_pr, color="black", linestyle="--", alpha=0.6, label=f"No-Skill Baseline ({baseline_pr:.1%})")
    axes[1].set_title("Precision-Recall (PR) Curve Comparison", fontweight="bold")
    axes[1].set_xlabel("Recall (Coverage of Cancellations)", fontweight="semibold")
    axes[1].set_ylabel("Precision (Accuracy of Cancellation Flags)", fontweight="semibold")
    axes[1].legend(loc="lower left", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig9_roc_pr_curves.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig9_roc_pr_curves.png"), dpi=300)
    plt.close(fig)
    print("[Evaluation] ROC and PR curves saved to reports/figures/ and dashboard/assets/")


def plot_calibration_curves(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray):
    """
    Plots Reliability Diagrams (Calibration Curves) to evaluate probability fidelity.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    selected_models = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Multi-Layer Perceptron (MLP)"]
    colors = ["#3B82F6", "#10B981", "#8B5CF6", "#EC4899", "#06B6D4"]
    
    for name, color in zip(selected_models, colors):
        if name in models_dict:
            model = models_dict[name]
            probs = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
            brier = brier_score_loss(y_test, probs)
            ax.plot(prob_pred, prob_true, marker="o", label=f"{name} (Brier: {brier:.4f})", color=color, linewidth=1.8)
            
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    ax.set_title("Probability Calibration Reliability Diagram", fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontweight="semibold")
    ax.set_ylabel("Empirical Fraction of Positives", fontweight="semibold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig10_calibration_curves.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig10_calibration_curves.png"), dpi=300)
    plt.close(fig)
    print("[Evaluation] Calibration curves saved.")


def run_cost_benefit_optimization(
    best_model, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    cost_fn: float = 150.0,   # Cost of False Negative: Unoccupied perishable room night revenue loss
    cost_fp: float = 40.0,    # Cost of False Positive: Relocation / discount friction
    gain_tp: float = 120.0,   # Value of True Positive: Salvaged room revenue via resale
    gain_tn: float = 150.0    # Value of True Negative: Normal occupied room revenue
) -> dict:
    """
    Computes expected financial profit curves across classification thresholds T in [0.05..0.95].
    Determines optimal decision threshold T* to maximize net business value.
    """
    probs = best_model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    
    profits = []
    savings_vs_default = []
    default_profit = None
    
    total_samples = len(y_test)
    
    for T in thresholds:
        preds = (probs >= T).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        
        # Expected Revenue Formulation:
        # TP: Resold / mitigated room revenue (+$120)
        # TN: Full room revenue (+$150)
        # FP: Overbooking friction / wasted discount (-$40)
        # FN: Lost perishable room night revenue (-$150)
        total_profit = (tp * gain_tp) + (tn * gain_tn) - (fp * cost_fp) - (fn * cost_fn)
        profit_per_booking = total_profit / total_samples
        profits.append(profit_per_booking)
        
        if round(T, 2) == 0.50:
            default_profit = profit_per_booking
            
    optimal_idx = np.argmax(profits)
    optimal_threshold = float(thresholds[optimal_idx])
    max_profit = float(profits[optimal_idx])
    
    # Financial metrics at default vs optimal
    preds_default = (probs >= 0.50).astype(int)
    tn_def, fp_def, fn_def, tp_def = confusion_matrix(y_test, preds_default).ravel()
    
    preds_opt = (probs >= optimal_threshold).astype(int)
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, preds_opt).ravel()
    
    net_improvement_per_booking = max_profit - (default_profit if default_profit else 0)
    annualized_lift_100k = net_improvement_per_booking * 100000
    
    cost_summary = {
        "cost_parameters": {
            "cost_false_negative_room_loss": cost_fn,
            "cost_false_positive_relocation": cost_fp,
            "gain_true_positive_resale": gain_tp,
            "gain_true_negative_stay": gain_tn
        },
        "default_threshold": 0.50,
        "default_profit_per_booking": round(float(default_profit), 2),
        "optimal_threshold": round(optimal_threshold, 2),
        "optimal_profit_per_booking": round(max_profit, 2),
        "profit_lift_per_booking": round(float(net_improvement_per_booking), 2),
        "annualized_gain_per_100k_bookings": round(float(annualized_lift_100k), 2),
        "confusion_matrix_default_0_5": {
            "tn": int(tn_def), "fp": int(fp_def), "fn": int(fn_def), "tp": int(tp_def)
        },
        "confusion_matrix_optimal": {
            "tn": int(tn_opt), "fp": int(fp_opt), "fn": int(fn_opt), "tp": int(tp_opt)
        }
    }
    
    # Plot Cost-Benefit Optimization Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Expected Value Curve vs Threshold
    axes[0].plot(thresholds, profits, color="#2563EB", linewidth=2.5, label="Net Revenue per Booking")
    axes[0].axvline(x=0.50, color="#6B7280", linestyle="--", label=f"Default T=0.50 (${default_profit:.2f}/booking)")
    axes[0].axvline(x=optimal_threshold, color="#10B981", linestyle="-", linewidth=2, label=f"Optimal T*={optimal_threshold:.2f} (${max_profit:.2f}/booking)")
    axes[0].set_title("Expected Net Revenue per Booking vs Decision Threshold", fontweight="bold")
    axes[0].set_xlabel("Probability Classification Threshold (T)", fontweight="semibold")
    axes[0].set_ylabel("Expected Net Revenue ($ / Booking)", fontweight="semibold")
    axes[0].legend(loc="lower center")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Right: Confusion Matrix Comparison at Optimal Threshold
    cm_opt_norm = np.array([[tn_opt, fp_opt], [fn_opt, tp_opt]])
    sns.heatmap(cm_opt_norm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=["Pred: Stay (0)", "Pred: Cancel (1)"],
                yticklabels=["Actual: Stay (0)", "Actual: Cancel (1)"])
    axes[1].set_title(f"Confusion Matrix at Optimal Cutoff (T* = {optimal_threshold:.2f})", fontweight="bold")
    axes[1].set_ylabel("True Ground Truth", fontweight="semibold")
    axes[1].set_xlabel("Operational Action", fontweight="semibold")
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig11_cost_benefit_optimization.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig11_cost_benefit_optimization.png"), dpi=300)
    plt.close(fig)
    
    # Save summary json
    with open(os.path.join(FIGURES_DIR, "cost_benefit_summary.json"), "w") as f:
        json.dump(cost_summary, f, indent=2)
    with open(os.path.join(ASSETS_DIR, "cost_benefit_summary.json"), "w") as f:
        json.dump(cost_summary, f, indent=2)
    # The decision engine reads its operational threshold from processed data.
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "cost_benefit_summary.json"), "w") as f:
        json.dump(cost_summary, f, indent=2)
        
    print(f"[Evaluation] Cost-Benefit Optimization complete. Optimal T* = {optimal_threshold:.2f} (Annualized Lift: ${annualized_lift_100k:,.2f})")
    return cost_summary


if __name__ == "__main__":
    pass
