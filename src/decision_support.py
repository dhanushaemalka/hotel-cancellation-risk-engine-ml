"""
Operational Decision Support System & Dynamic Overbooking Engine.
Domain 6: Tourism & Hospitality - Hotel Booking Demand
IT3091 Machine Learning Group Project
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "models")


class HotelDecisionEngine:
    """
    Production-ready decision support engine for hotel revenue managers.
    Provides real-time booking risk scoring, dynamic deposit rules, and 
    risk-constrained daily overbooking buffer optimization.
    """
    def __init__(self, model_name: str = "xgboost"):
        self.preprocessor_path = os.path.join(PROCESSED_DIR, "preprocessor.joblib")
        self.model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        self.personas_path = os.path.join(PROCESSED_DIR, "customer_personas.json")
        self.cost_summary_path = os.path.join(PROCESSED_DIR, "cost_benefit_summary.json")
        
        self.preprocessor = None
        self.model = None
        self.personas = {}
        self.optimal_threshold = 0.35
        
        self._load_artifacts()
        
    def _load_artifacts(self):
        if os.path.exists(self.preprocessor_path):
            self.preprocessor = joblib.load(self.preprocessor_path)
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.personas_path):
            with open(self.personas_path, "r") as f:
                self.personas = json.load(f)
        if os.path.exists(self.cost_summary_path):
            with open(self.cost_summary_path, "r") as f:
                c_data = json.load(f)
                self.optimal_threshold = c_data.get("optimal_threshold", 0.35)

    def predict_booking_risk(self, booking_dict: dict) -> dict:
        """
        Evaluates an individual booking dictionary and returns risk assessment,
        probability score, operational tier, and actionable recommendations.
        """
        # Convert dictionary to DataFrame
        df_single = pd.DataFrame([booking_dict])
        
        # Ensure engineered features are present
        if __package__:
            from .preprocessing import clean_raw_data, engineer_features
        else:
            from preprocessing import clean_raw_data, engineer_features
        df_clean = clean_raw_data(df_single) if "country" in df_single.columns else df_single.copy()
        df_feat = engineer_features(df_clean)
        
        # Transform via fitted preprocessor
        X_trans = self.preprocessor.transform(df_feat)
        
        # Predict probability
        prob_cancel = float(self.model.predict_proba(X_trans)[:, 1][0])
        
        # Assign Risk Tier
        if prob_cancel < 0.25:
            tier = "Low Risk"
            badge_color = "green"
            action = "Accept booking under standard flexible terms. High likelihood of arrival."
            deposit_req = "None (Standard Guarantee)"
        elif prob_cancel < 0.55:
            tier = "Medium Risk"
            badge_color = "amber"
            action = "Send automated 7-day pre-arrival confirmation SMS/Email with customized spa/dining add-ons to increase customer commitment."
            deposit_req = "Credit Card Pre-Authorization"
        else:
            tier = "High Risk"
            badge_color = "red"
            action = "Enforce 50% non-refundable deposit or require reconfirmation within 48 hours. Add room to dynamic overbooking pool."
            deposit_req = "Non-Refundable Deposit Required"
            
        is_above_optimal_threshold = bool(prob_cancel >= self.optimal_threshold)
        
        return {
            "cancellation_probability": round(prob_cancel, 3),
            "cancellation_probability_pct": round(prob_cancel * 100, 1),
            "optimal_threshold_applied": self.optimal_threshold,
            "action_trigger": "Mitigate / Overbook Resale" if is_above_optimal_threshold else "Standard Check-In Prep",
            "risk_tier": tier,
            "badge_color": badge_color,
            "recommended_deposit_policy": deposit_req,
            "operational_action": action
        }

    def calculate_overbooking_buffer(
        self, 
        cancellation_probabilities: list, 
        max_walk_risk_alpha: float = 0.005 # 0.5% maximum allowable walk probability
    ) -> dict:
        """
        Computes mathematically rigorous daily overbooking allowance using the 
        Normal approximation of the Poisson-Binomial distribution of cancellations.
        """
        probs = np.array(cancellation_probabilities)
        n_bookings = len(probs)
        
        # Expected cancellations and variance
        mu_cancellations = float(np.sum(probs))
        var_cancellations = float(np.sum(probs * (1.0 - probs)))
        std_cancellations = float(np.sqrt(max(var_cancellations, 1e-5)))
        
        # Critical value for one-sided risk tolerance (e.g., 99.5% confidence)
        z_crit = float(norm.ppf(1.0 - max_walk_risk_alpha))
        
        # Conservative buffer
        safe_buffer = max(0, int(np.floor(mu_cancellations - z_crit * std_cancellations)))
        aggressive_buffer = max(0, int(np.floor(mu_cancellations)))
        
        return {
            "total_bookings_analyzed": n_bookings,
            "expected_cancellations_mean": round(mu_cancellations, 1),
            "cancellations_std_dev": round(std_cancellations, 2),
            "risk_tolerance_alpha": max_walk_risk_alpha,
            "confidence_level_pct": round((1.0 - max_walk_risk_alpha) * 100, 1),
            "recommended_safe_overbooking_rooms": safe_buffer,
            "aggressive_overbooking_rooms": aggressive_buffer,
            "expected_occupancy_protection_pct": round((safe_buffer / max(mu_cancellations, 1)) * 100, 1)
        }


if __name__ == "__main__":
    pass
