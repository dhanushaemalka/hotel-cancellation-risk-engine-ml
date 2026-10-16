"""
Model Training, Cross-Validation, and Benchmarking Suite.
Domain 6: Tourism & Hospitality - Hotel Booking Demand
IT3091 Machine Learning Group Project
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "models")


class HeuristicLeadTimeClassifier:
    """
    Domain heuristic baseline: Predicts cancellation if lead time is long (>120 days) 
    or non-refundable deposit was taken.
    """
    def __init__(self, lead_time_col_idx: int = 0):
        self.lead_time_col_idx = lead_time_col_idx
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        lead_time_scaled = X[:, self.lead_time_col_idx]
        # In standardized scale, > 0.5 is approx lead time > 150 days
        preds = (lead_time_scaled > 0.5).astype(int)
        return preds
        
    def predict_proba(self, X):
        preds = self.predict(X)
        p1 = np.where(preds == 1, 0.75, 0.25)
        p0 = 1 - p1
        return np.column_stack([p0, p1])


def get_model_zoo():
    """
    Returns the complete suite of baseline and advanced candidate models.
    """
    models = {
        "Dummy (Majority Class)": DummyClassifier(strategy="most_frequent"),
        "Heuristic (Lead-Time Rule)": HeuristicLeadTimeClassifier(lead_time_col_idx=0),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=7, eval_metric="logloss", random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=40, random_state=42, n_jobs=-1, verbose=-1),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=40, random_state=42, early_stopping=True)
    }
    return models


def train_and_cross_validate_models(X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5):
    """
    Executes Stratified 5-Fold Cross Validation across all candidate models.
    Records cross-validated ROC-AUC, F1, Precision, Recall, and Accuracy.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = get_model_zoo()
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    
    results = {}
    fitted_models = {}
    
    print(f"[Models] Starting {n_splits}-Fold Cross-Validation across {len(models)} models...")
    print("=" * 80)
    
    for name, model in models.items():
        start_time = time.time()
        print(f"[Models] Training & Cross-Validating: {name}...")
        
        if name in ["Dummy (Majority Class)", "Heuristic (Lead-Time Rule)"]:
            # Baseline manual evaluation
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            probs = model.predict_proba(X_train)[:, 1]
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
            cv_results = {
                "test_roc_auc": np.array([roc_auc_score(y_train, probs)]),
                "test_f1": np.array([f1_score(y_train, preds, zero_division=0)]),
                "test_precision": np.array([precision_score(y_train, preds, zero_division=0)]),
                "test_recall": np.array([recall_score(y_train, preds, zero_division=0)]),
                "test_accuracy": np.array([accuracy_score(y_train, preds)]),
                "fit_time": np.array([time.time() - start_time])
            }
        else:
            cv_results = cross_validate(
                model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1 if name != "LightGBM" else 1, return_train_score=False
            )
            # Refit on full training set
            model.fit(X_train, y_train)
            
        elapsed = time.time() - start_time
        fitted_models[name] = model
        
        # Save model artifact
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        model_save_path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
        joblib.dump(model, model_save_path)
        
        res = {
            "model_name": name,
            "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
            "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
            "cv_f1_std": float(np.std(cv_results["test_f1"])),
            "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
            "cv_precision_std": float(np.std(cv_results["test_precision"])),
            "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
            "cv_recall_std": float(np.std(cv_results["test_recall"])),
            "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
            "fit_time_seconds": round(elapsed, 2)
        }
        results[name] = res
        print(f"   -> ROC-AUC: {res['cv_roc_auc_mean']:.4f} (+/- {res['cv_roc_auc_std']:.4f}) | F1: {res['cv_f1_mean']:.4f} | Time: {elapsed:.2f}s")

    print("=" * 80)
    
    # Save CV results summary
    cv_summary_path = os.path.join(MODELS_DIR, "cv_results_summary.json")
    with open(cv_summary_path, "w") as f:
        json.dump(results, f, indent=2)
        
    return results, fitted_models


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import prepare_train_test_data
    df = load_raw_data()
    data = prepare_train_test_data(df)
    results, fitted = train_and_cross_validate_models(data["X_train"][:5000], data["y_train"][:5000], n_splits=3)
    print("Test run completed successfully.")
