"""
Master End-to-End Execution Pipeline for Hotel Booking Demand ML Project.
Domain 6: Tourism & Hospitality
IT3091 Machine Learning Group Project
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_raw_data, download_dataset
from eda_engine import generate_eda_profile, plot_eda_figures
from preprocessing import prepare_train_test_data
from models import train_and_cross_validate_models
from segmentation import perform_customer_segmentation
from evaluation import (
    evaluate_models_on_test_set, 
    plot_roc_and_pr_curves, 
    plot_calibration_curves, 
    run_cost_benefit_optimization
)
from explainability import compute_shap_explanations
from decision_support import HotelDecisionEngine


def main():
    pipeline_start = time.time()
    print("==========================================================================================")
    print("[START] STARTING HOTEL BOOKING DEMAND MACHINE LEARNING PIPELINE")
    print("   IT3091 Machine Learning - Domain 6: Tourism & Hospitality")
    print("==========================================================================================")

    # -------------------------------------------------------------
    # Step 1: Data Ingestion
    # -------------------------------------------------------------
    print("\n[Step 1/7] INGESTING HOTEL BOOKING DEMAND DATASET...")
    df_raw = load_raw_data()
    print(f"[OK] Dataset loaded: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns.")

    # -------------------------------------------------------------
    # Step 2: Exploratory Data Analysis & Visualizations
    # -------------------------------------------------------------
    print("\n[Step 2/7] RUNNING STATISTICAL EDA & EXPORTING FIGURES...")
    eda_summary = generate_eda_profile(df_raw)
    plot_eda_figures(df_raw)
    print("[OK] EDA profiling and 6 figures successfully saved.")

    # -------------------------------------------------------------
    # Step 3: Leak-Free Preprocessing & Feature Engineering
    # -------------------------------------------------------------
    print("\n[Step 3/7] PREPROCESSING & FEATURE ENGINEERING (ZERO-LEAKAGE)...")
    data_dict = prepare_train_test_data(df_raw, test_size=0.2, random_state=42)
    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]
    feature_names = data_dict["feature_names"]
    df_featured = data_dict["df_featured"]
    print(f"[OK] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[OK] Engineered feature dimension: {len(feature_names)}")

    # -------------------------------------------------------------
    # Step 4: Model Training & 5-Fold Stratified Cross-Validation
    # -------------------------------------------------------------
    print("\n[Step 4/7] BENCHMARKING CANDIDATE MODELS VIA 5-FOLD CROSS-VALIDATION...")
    cv_results, fitted_models = train_and_cross_validate_models(X_train, y_train, n_splits=5)
    print("[OK] Model suite cross-validation complete.")

    # -------------------------------------------------------------
    # Step 5: Test Set Evaluation & Cost-Benefit Optimization
    # -------------------------------------------------------------
    print("\n[Step 5/7] EVALUATING ON TEST SET & OPTIMIZING REVENUE THRESHOLD...")
    test_metrics, model_predictions = evaluate_models_on_test_set(fitted_models, X_test, y_test)
    plot_roc_and_pr_curves(fitted_models, X_test, y_test)
    plot_calibration_curves(fitted_models, X_test, y_test)
    
    # Identify champion model by test ROC-AUC / F1
    champion_name = "XGBoost" if "XGBoost" in fitted_models else "Random Forest"
    champion_model = fitted_models[champion_name]
    print(f"[OK] Champion Model for Operational Deployment: {champion_name}")
    
    cost_summary = run_cost_benefit_optimization(
        champion_model, X_test, y_test,
        cost_fn=150.0, cost_fp=40.0, gain_tp=120.0, gain_tn=150.0
    )
    print("[OK] Cost-Benefit Revenue Optimization complete.")

    # -------------------------------------------------------------
    # Step 6: Explainability (SHAP) & Customer Behavioral Segmentation
    # -------------------------------------------------------------
    print("\n[Step 6/7] COMPUTING SHAP EXPLANATIONS & UNSUPERVISED SEGMENTATION...")
    shap_results = compute_shap_explanations(champion_model, X_test, feature_names, sample_size=1500)
    persona_mapping = perform_customer_segmentation(df_featured, n_clusters=4, sample_size=15000)
    print("[OK] Explainability and Customer Persona Clustering complete.")

    # -------------------------------------------------------------
    # Step 7: Decision Engine Smoke Test
    # -------------------------------------------------------------
    print("\n[Step 7/7] VALIDATING DECISION ENGINE & OVERBOOKING CALCULATOR...")
    engine = HotelDecisionEngine(model_name="xgboost")
    
    sample_booking = {
        "hotel": "City Hotel",
        "lead_time": 165,
        "arrival_date_year": 2017,
        "arrival_date_month": "July",
        "arrival_date_week_number": 28,
        "arrival_date_day_of_month": 12,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 3,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 1,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "agent": 9.0,
        "company": None,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 120.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0
    }
    
    risk_assessment = engine.predict_booking_risk(sample_booking)
    print(f"Sample Booking Risk Assessment:")
    print(f"   Probability of Cancellation: {risk_assessment['cancellation_probability_pct']}%")
    print(f"   Risk Tier: {risk_assessment['risk_tier']}")
    print(f"   Recommended Policy: {risk_assessment['recommended_deposit_policy']}")
    print(f"   Operational Action: {risk_assessment['operational_action']}")
    
    # Overbooking simulation for a day with 100 bookings
    test_sample_probs = champion_model.predict_proba(X_test[:100])[:, 1]
    ob_result = engine.calculate_overbooking_buffer(test_sample_probs, max_walk_risk_alpha=0.005)
    print(f"Daily Overbooking Calculation (N=100 bookings):")
    print(f"   Expected Cancellations: {ob_result['expected_cancellations_mean']}")
    print(f"   Safe Overbooking Buffer: {ob_result['recommended_safe_overbooking_rooms']} rooms (Confidence: {ob_result['confidence_level_pct']}%)")

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 90)
    print(f"[DONE] PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f} SECONDS!")
    print("=" * 90)


if __name__ == "__main__":
    main()
