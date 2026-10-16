"""
Preprocessing, Feature Engineering, and Leak-Free Transformation Pipeline.
Domain 6: Tourism & Hospitality - Hotel Booking Demand
IT3091 Machine Learning Group Project
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw hotel booking data by resolving anomalies and missingness.
    """
    df = df.copy()
    initial_shape = df.shape[0]

    # 1. Anomaly Filter: Remove rows with 0 total guests (impossible reservations)
    df["children"] = df["children"].fillna(0)
    total_guests = df["adults"] + df["children"] + df["babies"]
    df = df[total_guests > 0].copy()
    
    # 2. Anomaly Filter: Remove negative ADR and extreme ADR outliers (> $1000/night error)
    df = df[(df["adr"] >= 0) & (df["adr"] <= 1000)].copy()
    
    # 3. Handle Missing Values with domain-justified strategies
    df["country"] = df["country"].fillna("Unknown")
    
    # Consolidate top 20 countries, bucket rest to 'Other'
    top_countries = df["country"].value_counts().nlargest(20).index.tolist()
    df["country_grouped"] = df["country"].apply(lambda c: c if c in top_countries else "Other")
    
    # Agent and Company missingness indicates direct/private bookings
    df["is_agent_booking"] = df["agent"].notnull().astype(int)
    df["is_company_booking"] = df["company"].notnull().astype(int)
    
    # Clean categorical undefined entries
    df["meal"] = df["meal"].replace({"Undefined": "SC"})
    df["market_segment"] = df["market_segment"].replace({"Undefined": "Online TA"})
    df["distribution_channel"] = df["distribution_channel"].replace({"Undefined": "TA/TO"})
    
    cleaned_shape = df.shape[0]
    print(f"[Preprocessing] Cleaned dataset from {initial_shape:,} to {cleaned_shape:,} rows ({initial_shape - cleaned_shape:,} rows removed).")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs high-signal domain features while strictly avoiding target leakages.
    """
    df = df.copy()
    
    # 1. Total Stay Duration
    df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    
    # 2. Total Guest Count & Family Indicator
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    
    # 3. Room Type Mismatch (Operational risk indicator)
    df["room_type_mismatched"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    
    # 4. Pricing per guest
    df["adr_per_person"] = df["adr"] / (df["total_guests"] + 1e-5)
    
    # 5. Customer Historical Cancellation Ratio
    prev_total = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["cancellation_ratio"] = np.where(prev_total > 0, df["previous_cancellations"] / prev_total, 0.0)
    
    # 6. Customer Commitment Indicators
    df["has_special_requests"] = (df["total_of_special_requests"] > 0).astype(int)
    df["has_parking"] = (df["required_car_parking_spaces"] > 0).astype(int)
    df["has_booking_changes"] = (df["booking_changes"] > 0).astype(int)
    df["is_long_lead"] = (df["lead_time"] > 100).astype(int)
    
    # 7. Seasonality mapping
    season_map = {
        "December": "Winter", "January": "Winter", "February": "Winter",
        "March": "Spring", "April": "Spring", "May": "Spring",
        "June": "Summer", "July": "Summer", "August": "Summer",
        "September": "Autumn", "October": "Autumn", "November": "Autumn"
    }
    df["arrival_season"] = df["arrival_date_month"].map(season_map).fillna("Summer")
    
    # 8. Lead Time Bracket
    df["lead_time_bracket"] = pd.cut(
        df["lead_time"], 
        bins=[-1, 7, 30, 90, 180, 1000],
        labels=["0-7d", "8-30d", "31-90d", "91-180d", "180d+"]
    ).astype(str)
    
    # CRITICAL LEAKAGE PREVENTION:
    # Explicitly drop target-leaking attributes (reservation_status, reservation_status_date)
    leakage_cols = ["reservation_status", "reservation_status_date"]
    for col in leakage_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"[Preprocessing] Dropped TARGET LEAKAGE column: {col}")
            
    return df


def get_feature_lists():
    """
    Returns explicit lists of numerical and categorical feature columns for preprocessing pipelines.
    """
    numerical_features = [
        "lead_time",
        "total_stay",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "total_guests",
        "adr",
        "adr_per_person",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "cancellation_ratio",
        "booking_changes",
        "days_in_waiting_list",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "arrival_date_week_number",
        "arrival_date_day_of_month"
    ]
    
    categorical_features = [
        "hotel",
        "arrival_date_month",
        "arrival_season",
        "meal",
        "country_grouped",
        "market_segment",
        "distribution_channel",
        "is_repeated_guest",
        "reserved_room_type",
        "assigned_room_type",
        "deposit_type",
        "customer_type",
        "is_agent_booking",
        "is_company_booking",
        "room_type_mismatched",
        "is_family",
        "has_special_requests",
        "has_parking",
        "has_booking_changes",
        "is_long_lead",
        "lead_time_bracket"
    ]
    
    return numerical_features, categorical_features


def build_preprocessor(numerical_features, categorical_features):
    """
    Constructs a scikit-learn ColumnTransformer preprocessor with standard scaling and one-hot encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_train_test_data(df_raw: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Executes full leak-free data cleaning, feature engineering, and train/test splitting.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    df_clean = clean_raw_data(df_raw)
    df_featured = engineer_features(df_clean)
    
    target_col = "is_canceled"
    X = df_featured.drop(columns=[target_col])
    y = df_featured[target_col]
    
    # Stratified split to maintain cancellation ratio in train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"[Preprocessing] Split data into Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples.")
    print(f"[Preprocessing] Train cancellation rate: {y_train.mean():.2%}, Test cancellation rate: {y_test.mean():.2%}")
    
    num_cols, cat_cols = get_feature_lists()
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    # Fit preprocessor strictly on training data
    print("[Preprocessing] Fitting ColumnTransformer on X_train...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # Get feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(cat_feature_names)
    
    # Save preprocessor artifact
    preprocessor_path = os.path.join(PROCESSED_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save feature names list
    with open(os.path.join(PROCESSED_DIR, "feature_names.json"), "w") as f:
        json.dump(all_feature_names, f, indent=2)
        
    # Save clean featured datasets
    try:
        df_featured.to_parquet(os.path.join(PROCESSED_DIR, "hotel_bookings_featured.parquet"))
    except Exception:
        df_featured.to_csv(os.path.join(PROCESSED_DIR, "hotel_bookings_featured.csv.gz"), index=False, compression="gzip")
    
    print(f"[Preprocessing] Transformed feature space: {len(all_feature_names)} features ({len(num_cols)} numerical, {len(cat_feature_names)} one-hot encoded).")
    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train": X_train_trans,
        "X_test": X_test_trans,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names": all_feature_names,
        "preprocessor": preprocessor,
        "df_featured": df_featured
    }


if __name__ == "__main__":
    from data_loader import load_raw_data
    df = load_raw_data()
    data = prepare_train_test_data(df)
    print("Preprocessing completed successfully.")
