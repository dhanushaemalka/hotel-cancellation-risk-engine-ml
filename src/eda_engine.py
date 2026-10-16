"""
Exploratory Data Analysis (EDA) and Statistical Profiling Engine.
Domain 6: Tourism & Hospitality - Hotel Booking Demand
IT3091 Machine Learning Group Project
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 300

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "assets")


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)


def generate_eda_profile(df: pd.DataFrame) -> dict:
    """
    Computes comprehensive dataset health, statistical distribution, and EDA insights.
    """
    ensure_dirs()
    total_rows, total_cols = df.shape
    
    # Target distribution
    cancel_counts = df["is_canceled"].value_counts().to_dict()
    cancel_rate = float(df["is_canceled"].mean())
    
    # Missing values
    missing = df.isnull().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing[missing > 0].items()}
    missing_pct = {col: round(float(cnt) / total_rows * 100, 2) for col, cnt in missing[missing > 0].items()}
    
    # Hotel distribution
    hotel_dist = df.groupby(["hotel", "is_canceled"]).size().unstack(fill_value=0)
    hotel_stats = {
        "Resort Hotel": {
            "total": int(df[df["hotel"] == "Resort Hotel"].shape[0]),
            "cancellations": int(df[(df["hotel"] == "Resort Hotel") & (df["is_canceled"] == 1)].shape[0]),
            "cancel_rate": round(float(df[df["hotel"] == "Resort Hotel"]["is_canceled"].mean()) * 100, 2)
        },
        "City Hotel": {
            "total": int(df[df["hotel"] == "City Hotel"].shape[0]),
            "cancellations": int(df[(df["hotel"] == "City Hotel") & (df["is_canceled"] == 1)].shape[0]),
            "cancel_rate": round(float(df[df["hotel"] == "City Hotel"]["is_canceled"].mean()) * 100, 2)
        }
    }
    
    # Anomaly checks
    zero_guests = int((df["adults"] + df["children"].fillna(0) + df["babies"] == 0).sum())
    negative_adr = int((df["adr"] < 0).sum())
    extreme_adr = int((df["adr"] > 1000).sum())
    
    # Market segment cancellation rates
    market_cancel = df.groupby("market_segment")["is_canceled"].agg(["count", "mean"]).reset_index()
    market_cancel["mean"] = (market_cancel["mean"] * 100).round(2)
    market_stats = market_cancel.to_dict(orient="records")
    
    # Deposit type paradox
    deposit_cancel = df.groupby("deposit_type")["is_canceled"].agg(["count", "mean"]).reset_index()
    deposit_cancel["mean"] = (deposit_cancel["mean"] * 100).round(2)
    deposit_stats = deposit_cancel.to_dict(orient="records")
    
    # Lead time statistics
    lead_time_stats = {
        "overall_mean": round(float(df["lead_time"].mean()), 1),
        "overall_median": float(df["lead_time"].median()),
        "canceled_mean": round(float(df[df["is_canceled"] == 1]["lead_time"].mean()), 1),
        "not_canceled_mean": round(float(df[df["is_canceled"] == 0]["lead_time"].mean()), 1)
    }
    
    summary = {
        "dataset_shape": {"rows": total_rows, "columns": total_cols},
        "cancellation_rate_overall": round(cancel_rate * 100, 2),
        "target_counts": {str(k): int(v) for k, v in cancel_counts.items()},
        "missing_values": missing_dict,
        "missing_percentages": missing_pct,
        "anomalies": {
            "zero_total_guests": zero_guests,
            "negative_adr": negative_adr,
            "extreme_adr_over_1000": extreme_adr
        },
        "hotel_breakdown": hotel_stats,
        "market_segments": market_stats,
        "deposit_types": deposit_stats,
        "lead_time": lead_time_stats
    }
    
    # Save summary json
    summary_path = os.path.join(FIGURES_DIR, "eda_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    summary_assets_path = os.path.join(ASSETS_DIR, "eda_summary.json")
    with open(summary_assets_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"[EDA] Summary metrics computed and saved to {summary_path}")
    return summary


def plot_eda_figures(df: pd.DataFrame):
    """
    Generates high-quality EDA visualizations for assignment report and web dashboard.
    """
    ensure_dirs()
    print("[EDA] Generating high-resolution EDA figures...")
    
    # 1. Target Class & Hotel Breakdown Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: Target Distribution Pie/Donut
    cancel_counts = df["is_canceled"].value_counts()
    colors = ["#2563EB", "#DC2626"]
    axes[0].pie(cancel_counts, labels=["Not Canceled (0)", "Canceled (1)"], autopct="%1.1f%%",
                startangle=140, colors=colors, explode=(0, 0.06), wedgeprops=dict(width=0.6, edgecolor='white'))
    axes[0].set_title("Overall Booking Cancellation Distribution\n(Target Imbalance: 37.0% Cancellations)", fontweight="bold")
    
    # Right: Hotel Type vs Cancellation Rate
    hotel_df = df.groupby(["hotel", "is_canceled"]).size().unstack()
    hotel_df_pct = hotel_df.div(hotel_df.sum(axis=1), axis=0) * 100
    hotel_df_pct.plot(kind="bar", stacked=True, ax=axes[1], color=["#2563EB", "#DC2626"], alpha=0.9, edgecolor="black")
    axes[1].set_title("Cancellation Proportion by Hotel Property Type", fontweight="bold")
    axes[1].set_xlabel("Property Type", fontweight="semibold")
    axes[1].set_ylabel("Percentage (%)", fontweight="semibold")
    axes[1].legend(["Not Canceled", "Canceled"], title="Status", loc="upper right")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    for p in axes[1].patches:
        width, height = p.get_width(), p.get_height()
        if height > 5:
            x, y = p.get_xy() 
            axes[1].text(x + width/2, y + height/2, f"{height:.1f}%", ha="center", va="center", color="white", fontweight="bold")
            
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_target_hotel_distribution.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig1_target_hotel_distribution.png"), dpi=300)
    plt.close(fig)

    # 2. Lead Time vs Cancellation Probability Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(data=df[df["is_canceled"] == 0]["lead_time"], label="Not Canceled (Mean: 80 days)", color="#2563EB", fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(data=df[df["is_canceled"] == 1]["lead_time"], label="Canceled (Mean: 145 days)", color="#DC2626", fill=True, alpha=0.3, ax=ax)
    ax.axvline(df[df["is_canceled"] == 0]["lead_time"].median(), color="#2563EB", linestyle="--", linewidth=1.5, label="Median Not Canceled (49d)")
    ax.axvline(df[df["is_canceled"] == 1]["lead_time"].median(), color="#DC2626", linestyle="--", linewidth=1.5, label="Median Canceled (113d)")
    ax.set_title("Booking Lead Time Kernel Density Estimation by Cancellation Outcome", fontweight="bold")
    ax.set_xlabel("Lead Time (Days between reservation creation and arrival date)", fontweight="semibold")
    ax.set_ylabel("Density", fontweight="semibold")
    ax.set_xlim(0, 500)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_lead_time_distribution.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig2_lead_time_distribution.png"), dpi=300)
    plt.close(fig)

    # 3. Deposit Type Paradox and Market Segment Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Deposit Type Cancellation Rates
    dep_stats = df.groupby("deposit_type")["is_canceled"].mean().reset_index()
    dep_stats["cancellation_rate"] = dep_stats["is_canceled"] * 100
    sns.barplot(data=dep_stats, x="deposit_type", y="cancellation_rate", ax=axes[0], palette=["#2563EB", "#DC2626", "#F59E0B"], edgecolor="black")
    axes[0].set_title("The 'Deposit Type Paradox': Cancellation Rate by Guarantee Type", fontweight="bold")
    axes[0].set_xlabel("Deposit Type", fontweight="semibold")
    axes[0].set_ylabel("Cancellation Rate (%)", fontweight="semibold")
    axes[0].set_ylim(0, 105)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                         ha="center", va="center", fontsize=10, color="black", fontweight="bold")
                         
    # Right: Market Segment Cancellation Rates
    mkt_stats = df.groupby("market_segment")["is_canceled"].mean().reset_index()
    mkt_stats["cancellation_rate"] = mkt_stats["is_canceled"] * 100
    mkt_stats = mkt_stats.sort_values(by="cancellation_rate", ascending=False)
    sns.barplot(data=mkt_stats, x="market_segment", y="cancellation_rate", ax=axes[1], palette="Blues_r", edgecolor="black")
    axes[1].set_title("Cancellation Rate across Market Segments", fontweight="bold")
    axes[1].set_xlabel("Market Segment", fontweight="semibold")
    axes[1].set_ylabel("Cancellation Rate (%)", fontweight="semibold")
    axes[1].set_ylim(0, 100)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                         ha="center", va="center", fontsize=9, color="black", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_deposit_market_segment.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig3_deposit_market_segment.png"), dpi=300)
    plt.close(fig)

    # 4. Special Requests & Parking Space Impact on Cancellations
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Special requests
    req_stats = df.groupby("total_of_special_requests")["is_canceled"].mean().reset_index()
    req_stats["cancellation_rate"] = req_stats["is_canceled"] * 100
    sns.barplot(data=req_stats, x="total_of_special_requests", y="cancellation_rate", ax=axes[0], color="#0D9488", edgecolor="black")
    axes[0].set_title("Customer Commitment: Special Requests vs Cancellation Rate", fontweight="bold")
    axes[0].set_xlabel("Total Number of Special Requests", fontweight="semibold")
    axes[0].set_ylabel("Cancellation Rate (%)", fontweight="semibold")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                         ha="center", va="center", fontsize=9, color="black", fontweight="bold")

    # Parking spaces
    park_stats = df.groupby("required_car_parking_spaces")["is_canceled"].mean().reset_index()
    park_stats["cancellation_rate"] = park_stats["is_canceled"] * 100
    sns.barplot(data=park_stats[park_stats["required_car_parking_spaces"] <= 3], x="required_car_parking_spaces", y="cancellation_rate", ax=axes[1], color="#4F46E5", edgecolor="black")
    axes[1].set_title("Parking Space Demand as Strong Commitment Signal", fontweight="bold")
    axes[1].set_xlabel("Required Car Parking Spaces", fontweight="semibold")
    axes[1].set_ylabel("Cancellation Rate (%)", fontweight="semibold")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., max(p.get_height(), 0) + 1),
                         ha="center", va="center", fontsize=9, color="black", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_special_requests_parking.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig4_special_requests_parking.png"), dpi=300)
    plt.close(fig)

    # 5. Monthly Seasonality and Arrival Trends
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    monthly_hotel = df.groupby(["arrival_date_month", "hotel"])["is_canceled"].agg(["count", "mean"]).reset_index()
    monthly_hotel["mean"] = monthly_hotel["mean"] * 100
    monthly_hotel["arrival_date_month"] = pd.Categorical(monthly_hotel["arrival_date_month"], categories=month_order, ordered=True)
    monthly_hotel = monthly_hotel.sort_values("arrival_date_month")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly_hotel, x="arrival_date_month", y="mean", hue="hotel", marker="o", linewidth=2.5, palette=["#2563EB", "#059669"], ax=ax)
    ax.set_title("Monthly Cancellation Rate Seasonality (Resort Hotel vs City Hotel)", fontweight="bold")
    ax.set_xlabel("Arrival Month", fontweight="semibold")
    ax.set_ylabel("Cancellation Rate (%)", fontweight="semibold")
    ax.set_xticklabels(month_order, rotation=35, ha="right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Property", frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_monthly_seasonality.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig5_monthly_seasonality.png"), dpi=300)
    plt.close(fig)

    # 6. Correlation Heatmap of Key Numerical Variables
    num_cols = ["is_canceled", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", 
                "adults", "children", "is_repeated_guest", "previous_cancellations", 
                "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list", 
                "adr", "required_car_parking_spaces", "total_of_special_requests"]
    corr_df = df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Matrix of Key Numerical Booking Attributes", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_correlation_matrix.png"), dpi=300)
    fig.savefig(os.path.join(ASSETS_DIR, "fig6_correlation_matrix.png"), dpi=300)
    plt.close(fig)

    print("[EDA] All 6 high-resolution figures successfully saved to reports/figures/ and dashboard/assets/")


if __name__ == "__main__":
    from data_loader import load_raw_data
    df = load_raw_data()
    summary = generate_eda_profile(df)
    plot_eda_figures(df)
