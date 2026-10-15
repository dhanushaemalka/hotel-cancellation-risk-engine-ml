# Hotel Booking Demand Intelligence & Revenue Risk Mitigation System
### IT3091 Machine Learning Group Project — Domain 6: Tourism & Hospitality

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple.svg)](https://shap.readthedocs.io/)

---

## 📖 Project Overview
This repository contains the complete, production-grade Machine Learning and Decision Support solution for **Domain 6: Tourism & Hospitality — Hotel Booking Demand** under the **IT3091 Machine Learning** assignment descriptor.

Hotels face massive revenue destruction from booking cancellations (baseline **37.08%** cancellation rate across 119,390 reservations). This system integrates:
1. **Primary Lens (Supervised Classification)**: Predicts cancellation probability $P(\text{is\_canceled}=1)$ and applies an empirical **Expected Value Cost Matrix** ($C_{FN}=\$150$ room loss vs $C_{FP}=\$40$ overbooking friction) to optimize the operational threshold **$T^* = 0.39$**, delivering **+$84,557** annualized net profit per 100k bookings.
2. **Secondary Lens (Unsupervised Segmentation)**: Clusters guests via $K$-Means ($K=4$) into actionable personas (*Spontaneous Business*, *High-Risk Long-Lead*, *Premium Family*, *Loyal Repeat*).
3. **Interactive Executive Web Dashboard**: A dark glassmorphic dashboard for real-time booking risk scoring, overbooking calculations, and scenario simulation.

---

## 📂 Repository Structure
```
c:\Users\dhanu\Desktop\ml\
├── data/
│   ├── raw/
│   │   └── hotel_bookings.csv             # Raw PMS dataset (119,390 rows x 32 columns)
│   └── processed/
│       ├── hotel_bookings_featured.parquet# Clean, featured dataset (zero leakage)
│       ├── preprocessor.joblib            # Fitted ColumnTransformer (scaling + OHE)
│       ├── customer_personas.json         # 4 Customer persona profiles
│       └── models/                        # Serialized champion models (.joblib)
├── src/
│   ├── data_loader.py                     # Dataset fetcher & Data Dictionary provider
│   ├── eda_engine.py                      # Statistical profiler & 6 publication figures
│   ├── preprocessing.py                   # Leakage audit & feature engineering pipeline
│   ├── models.py                          # 7 Candidate models & 5-Fold Stratified CV
│   ├── evaluation.py                      # ROC/PR curves, Brier calibration & Cost Matrix
│   ├── explainability.py                  # SHAP beeswarm & feature importance ranking
│   ├── segmentation.py                    # K-Means clustering, Elbow, Silhouette & PCA
│   └── decision_support.py                # Production Decision Support & Overbooking Engine
├── notebooks/
│   └── hotel_booking_demand_master.ipynb  # End-to-end reproducible Jupyter Notebook
├── reports/
│   ├── FINAL_PROJECT_REPORT.md            # Comprehensive 100-mark report with all 6 logs
│   ├── INITIAL_SUBMISSION_PROPOSAL.md     # Initial submission brief & role matrix
│   ├── DATA_DICTIONARY.md                 # Complete 32-attribute reference
│   ├── PERSONAL_LEARNING_JOURNEY.md       # Individual reflective journal template
│   └── figures/                           # High-resolution exported figures (fig1-fig13)
├── dashboard/
│   ├── index.html                         # Interactive executive web dashboard
│   ├── styles.css                         # Dark glassmorphic design system
│   ├── app.js                             # Live risk predictor & overbooking simulator
│   └── assets/                            # Exported charts and JSON summaries
├── run_pipeline.py                        # Single CLI command to run entire ML pipeline
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Clone or open this repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute the Full ML Pipeline
Run the complete end-to-end pipeline (Data Ingestion $\to$ EDA $\to$ Preprocessing $\to$ 5-Fold CV $\to$ Test Evaluation $\to$ Cost Matrix $\to$ SHAP $\to$ Segmentation):
```bash
python run_pipeline.py
```

### 3. Launch Interactive Decision Support Dashboard
Open `dashboard/index.html` directly in your browser, or serve it locally:
```bash
python -m http.server 8000 --directory dashboard
```
Then navigate to `http://localhost:8000` to interact with the live booking predictor and overbooking engine.

### 4. Explore Jupyter Notebook
Open the interactive Jupyter Notebook:
```bash
jupyter notebook notebooks/hotel_booking_demand_master.ipynb
```

---

## 📊 Key Results Summary

| Model Candidate | 5-Fold CV ROC-AUC | Test Accuracy | Test F1-Score | Test ROC-AUC | Test PR-AUC | Test Brier Score |
|---|---|---|---|---|---|---|
| **Dummy Baseline** | 0.5000 | 62.92% | 0.0000 | 0.5000 | 0.3708 | 0.3708 |
| **Heuristic Baseline** | 0.5988 | 65.50% | 0.4521 | 0.5993 | 0.4395 | 0.2350 |
| **Logistic Regression** | 0.9097 | 83.20% | 0.7566 | 0.9096 | 0.8764 | 0.1146 |
| **Random Forest** | **0.9459** | **87.40%** | **0.8317** | **0.9473** | **0.9239** | 0.0933 |
| **XGBoost (Champion)** | 0.9423 | 86.75% | 0.8144 | 0.9427 | 0.9173 | 0.0926 |
| **LightGBM** | 0.9437 | 86.99% | 0.8176 | 0.9453 | 0.9202 | **0.0907** |
| **Multi-Layer Perceptron** | 0.9386 | 86.47% | 0.8152 | 0.9398 | 0.9139 | 0.0948 |

---

## 📑 Core Evidence Logs for IT3091 Assignment
1. **Problem Framing Canvas**: In [`reports/FINAL_PROJECT_REPORT.md`](reports/FINAL_PROJECT_REPORT.md#11-problem-framing-canvas)
2. **8-Point Architectural Decision Log**: In [`reports/FINAL_PROJECT_REPORT.md`](reports/FINAL_PROJECT_REPORT.md#22-architectural-decision-log)
3. **Data Dictionary & Quality Profiling**: In [`reports/DATA_DICTIONARY.md`](reports/DATA_DICTIONARY.md)
4. **EDA Insight Log**: In [`reports/FINAL_PROJECT_REPORT.md`](reports/FINAL_PROJECT_REPORT.md#32-structured-eda-insight-log)
5. **Preprocessing & Feature Engineering Log**: In [`reports/FINAL_PROJECT_REPORT.md`](reports/FINAL_PROJECT_REPORT.md#4-leak-free-preprocessing--feature-engineering-log-15-marks)
6. **Model Comparison Matrix & Evaluation**: In [`reports/FINAL_PROJECT_REPORT.md`](reports/FINAL_PROJECT_REPORT.md#52-multi-metric-model-comparison-matrix)
7. **Recommendations & Limitations Playbook**: In [`reports/FINAL_PROJECT_REPORT.md`](reports/FINAL_PROJECT_REPORT.md#8-recommendations-limitations--responsible-ai-playbook-10-marks)
8. **Initial Proposal Submission Brief**: In [`reports/INITIAL_SUBMISSION_PROPOSAL.md`](reports/INITIAL_SUBMISSION_PROPOSAL.md)
9. **Individual Learning Journey Reflection**: In [`reports/PERSONAL_LEARNING_JOURNEY.md`](reports/PERSONAL_LEARNING_JOURNEY.md)
